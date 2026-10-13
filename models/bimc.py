import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
from pathlib import Path

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class BiMC(nn.Module):

    def __init__(self, cfg, template, device):
        super(BiMC, self).__init__()
        self.cfg = cfg
        self.device = device
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print(f"Prompt template:{template}")
        self.template = template
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BiMC.PREC == "fp32" or cfg.TRAINER.BiMC.PREC == "amp":
        # CLIP's default precision is fp16
            clip_model.float()

        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.text_proto = None
        self.description_proto = None
        self.vision_proto = None


    @torch.no_grad()
    def inference_text_feature(self, class_names, template, cls_begin_index):
        print(f'class names: {class_names}')
        clip_weights = []
        all_targets = []
        k = cls_begin_index
        for classname in class_names:
            targets = torch.full((len(template),), k)
            all_targets.append(targets)
            k += 1
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            classname = classname.replace('-', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=0)
        clip_weights = F.normalize(clip_weights, dim=-1)
        all_targets = torch.cat(all_targets, dim=0)
        return clip_weights, all_targets


    @torch.no_grad()
    def inference_all_img_feature(self, loader, cls_begin_index):
        all_features = []
        all_labels = []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features = self.clip_model.encode_image(images)
            features = F.normalize(features, dim=-1)
            all_features.append(features)
            all_labels.append(labels)
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        unique_labels = torch.unique(all_labels)
        print(f'all targets:{unique_labels}')
        prototypes = []
        for c in unique_labels:
            idx = torch.where(c == all_labels)[0]
            class_features = all_features[idx]
            class_prototype = class_features.mean(dim=0)
            prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, dim=-1)
        return all_features, all_labels, prototypes


    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, cls_begin_index):
        description_embeddings = []
        mean_embeddings = []
        all_targets = []
        file = open(gpt_path, "r")
        GPT_prompt_dict = json.load(file)
        # The order of embeddings should follow strictly order of classname variable
        # Keys name should match classnames so that we could do fetching from the dict.
        # Convert the dict to lower case
        GPT_prompt_dict = {k.lower().replace("_", " "): v for k, v in GPT_prompt_dict.items()}
        k = cls_begin_index
        for single_key in class_names:
            single_class_prompts = GPT_prompt_dict[single_key.lower().replace("_", " ")]
            targets = torch.full((len(single_class_prompts),), k)

            k += 1
            x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
            with torch.no_grad():
                text_features = self.clip_model.encode_text(x_tokenized.cuda())
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_embeddings.append(text_features.mean(0).unsqueeze(0))
            description_embeddings.append(text_features)
            all_targets.append(targets)
        description_embeddings = torch.cat(description_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mean_embeddings = torch.cat(mean_embeddings, dim=0)
        mean_embeddings = F.normalize(mean_embeddings, dim=-1)
        return description_embeddings, all_targets, mean_embeddings



    @torch.no_grad()
    def inference_image_description_prototype(self, class_names, loader, class_index,
                                              description_path, max_per_class=5):
        """CLIP description prototypes using ONLY current-session training support.

        Inputs:
            class_names: [C_task] names in the order of class_index.
            loader: task training data loader, with dataset.source_indices.
            class_index: [C_task] zero-based global class labels.
            description_path: CUB/AREA-style directory or image-ID JSON file.
            max_per_class: captions per class, 0 means all support examples.
        Returns:
            prototypes: [C_task,D] normalized text prototypes.
            counts: [C_task] selected caption counts.
        """
        path = Path(description_path)
        if not description_path or not path.exists():
            raise FileNotFoundError(
                f"IMAGE_DESCRIPTION_PATH not found: {description_path!r}. "
                "Provide captions for the selected training images first."
            )
        if max_per_class < 0:
            raise ValueError("IMAGE_DESCRIPTION_MAX_PER_CLASS must be >= 0.")
        support = loader.dataset
        if not hasattr(support, "source_indices"):
            raise ValueError("Training dataset must expose original source_indices.")
        if len(class_names) != len(class_index):
            raise ValueError("class_names and class_index length mismatch.")

        by_image_id = None
        if path.is_file():
            if path.suffix.lower() != ".json":
                raise ValueError("Image caption file must be JSON, or use CUB directory.")
            with path.open("r", encoding="utf-8") as handle:
                by_image_id = json.load(handle)
            if not isinstance(by_image_id, dict):
                raise ValueError("Caption JSON must map image IDs to strings.")
        elif not path.is_dir():
            raise ValueError(f"Invalid image caption path: {path}")

        prototypes, counts = [], []
        for class_name, global_label in zip(class_names, class_index):
            global_label = int(global_label)
            indices = [i for i, target in enumerate(support.labels)
                       if int(target) == global_label]
            if not indices:
                raise ValueError(f"No support images for class {global_label}: {class_name}")
            if max_per_class:
                indices = indices[:max_per_class]

            captions, missing = [], []
            if by_image_id is None:
                matches = sorted(path.glob(f"{global_label + 1:03d}.*_descriptions.txt"))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected one CUB caption file for class {global_label} "
                        f"({class_name}) in {path}; found {len(matches)}"
                    )
                class_captions = {}
                with matches[0].open("r", encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        parts = line.rstrip("\n").split("\t", 1)
                        if len(parts) != 2 or not parts[1].strip():
                            raise ValueError(f"Invalid caption line {matches[0]}:{line_no}")
                        image_name = Path(parts[0]).name
                        if image_name in class_captions:
                            raise ValueError(f"Duplicate caption for {image_name} in {matches[0]}")
                        class_captions[image_name] = parts[1].strip()

            for i in indices:
                img = support.images[i]
                image_id = (Path(str(img)).name if support.use_path
                            else str(int(support.source_indices[i])))
                caption = (by_image_id if by_image_id is not None else class_captions).get(image_id)
                if not isinstance(caption, str) or not caption.strip():
                    missing.append(image_id)
                else:
                    captions.append(caption.strip())
            if missing:
                raise ValueError(
                    f"Missing {len(missing)} support-image captions for class "
                    f"{global_label} ({class_name}), e.g. {missing[:5]}. "
                    "Do not replace them using test or future-session images."
                )

            encoded = []
            for start in range(0, len(captions), 32):
                # AREA image captions can exceed the CLIP 77-token limit.
                tokens = clip.tokenize(captions[start:start + 32], truncate=True).to(self.device)
                features = self.clip_model.encode_text(tokens).float()  # [B_caption,D]
                encoded.append(F.normalize(features, dim=-1))
            class_features = torch.cat(encoded, dim=0)  # [N_caption,D]
            prototypes.append(F.normalize(class_features.mean(dim=0), dim=0))  # [D]
            counts.append(len(captions))
        return torch.stack(prototypes, dim=0), counts  # [C_task,D], [C_task]


    @torch.no_grad()
    def compute_attribute_basis(self, features, labels, K=8):
        """Compute per-class PGA-like tangent attribute bases from CLIP features.

        Inputs:
            features: [N, D] L2-normalized or raw image embeddings.
            labels: [N] integer global class IDs.
            K: number of attribute directions per class.
        Returns:
            basis: [C, D, K], padded with zeros if a class has fewer directions.
            centers: [C, D] normalized class means (extrinsic Frechet approximation).
        The C rows follow torch.unique(labels, sorted=True).
        """
        x_all = F.normalize(features.float(), dim=-1)  # [N, D]
        basis_list, center_list = [], []
        for label in torch.unique(labels, sorted=True):
            x = x_all[labels == label]                    # [N_c, D]
            center = F.normalize(x.mean(dim=0), dim=0)   # [D]
            cosine = (x @ center).clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # [N_c]
            theta = torch.acos(cosine)                   # [N_c]
            scale = theta / torch.sqrt((1 - cosine.square()).clamp_min(1e-12))
            tangent = scale[:, None] * (x - cosine[:, None] * center[None, :])  # [N_c, D]
            # Right singular vectors span the leading tangent directions.
            _, _, vh = torch.linalg.svd(tangent, full_matrices=False)
            rank = min(K, vh.shape[0], max(x.shape[0] - 1, 0))
            directions = vh[:rank].T                     # [D, rank]
            if rank < K:
                directions = F.pad(directions, (0, K - rank))  # [D, K]
            basis_list.append(directions)
            center_list.append(center)
        return torch.stack(basis_list, dim=0), torch.stack(center_list, dim=0)

    @torch.no_grad()
    def attribute_logits(self, img_feat, basis, centers):
        """Negative tangent-subspace reconstruction error for each class.

        Inputs:
            img_feat: [B, D] CLIP image features.
            basis: [C, D, K] class-specific tangent directions.
            centers: [C, D] original normalized class centers.
        Returns:
            scores: [B, C], larger means closer to the attribute subspace.
        """
        z = F.normalize(img_feat.float(), dim=-1)     # [B, D]
        mu = F.normalize(centers.float(), dim=-1)     # [C, D]
        basis = basis.float()                          # [C, D, K]
        cosine = (z @ mu.T).clamp(-1 + 1e-6, 1 - 1e-6)  # [B, C]
        theta = torch.acos(cosine)                    # [B, C]
        scale = theta / torch.sqrt((1 - cosine.square()).clamp_min(1e-12))
        tangent = scale[..., None] * (
            z[:, None, :] - cosine[..., None] * mu[None, :, :]
        )                                             # [B, C, D]
        coeff = torch.einsum("bcd,cdk->bck", tangent, basis)  # [B, C, K]
        projection = torch.einsum("bck,cdk->bcd", coeff, basis)  # [B, C, D]
        residual = tangent - projection               # [B, C, D]
        # Experiment A: normalize by total tangent energy to reduce
        # dependence on the query-to-class-center geodesic distance.
        residual_energy = residual.square().sum(dim=-1)  # [B, C]
        total_energy = tangent.square().sum(dim=-1)      # [B, C]
        scores = -residual_energy / total_energy.clamp_min(1e-6)  # [B, C]
        return scores  # [B, C]

    def soft_calibration(self, base_protos, cur_protos):
        shift_weight = self.cfg.TRAINER.BiMC.LAMBDA_I
        tau = self.cfg.TRAINER.BiMC.TAU
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        weights = torch.mm(cur_protos, base_protos.T) * tau
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        updated_protos = (1 - shift_weight) * cur_protos + shift_weight * delta_protos
        updated_protos = F.normalize(updated_protos, dim=-1)
        return updated_protos
    

    def build_task_statistics(self, class_names, loader,
                         class_index, calibrate_novel_vision_proto=False):
        
            
        def shrink_cov(cov, alpha1=1.0, alpha2=0.0):
            diag_mean = torch.mean(torch.diagonal(cov))
            off_diag = cov.clone()
            off_diag.fill_diagonal_(0.0)
            mask = off_diag != 0.0
            off_diag_mean = (off_diag*mask).sum() / mask.sum()
            iden = torch.eye(cov.shape[0]).to(cov.device)
            cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
            return cov_


        cls_begin_index = class_index[0]


        text_features, text_targets = self.inference_text_feature(class_names, self.template, cls_begin_index)

        # Keep label-generated descriptions for the optional KNN ensemble head.
        description_features, description_targets, description_proto = \
            self.inference_all_description_feature(
                class_names=class_names, gpt_path=self.cfg.DATASET.GPT_PATH,
                cls_begin_index=cls_begin_index)

        # Image captions replace ONLY description_proto in text calibration:
        # (1-lambda_t)*text_features + lambda_t*description_proto.
        image_description_counts = None
        source = self.cfg.DATASET.DESCRIPTION_SOURCE.lower()
        if source == "image":
            description_proto, image_description_counts = \
                self.inference_image_description_prototype(
                    class_names=class_names, loader=loader,
                    class_index=class_index,
                    description_path=self.cfg.DATASET.IMAGE_DESCRIPTION_PATH,
                    max_per_class=int(self.cfg.DATASET.IMAGE_DESCRIPTION_MAX_PER_CLASS))
            print(f"Image-caption text calibration: {image_description_counts} captions per class")
        elif source != "label":
            raise ValueError(f"Unexpected DESCRIPTION_SOURCE={source!r}; use label or image.")

        images_features, images_targets, images_proto = \
                                    self.inference_all_img_feature(loader, cls_begin_index)

        # [C_task, D, K] attribute bases, [C_task, D] tangent-space centers.
        # Build from original normalized features BEFORE vision prototype calibration.
        use_pga = getattr(self.cfg.TRAINER.BiMC, 'USE_PGA', False)
        if use_pga:
            pga_basis, pga_centers = self.compute_attribute_basis(
                images_features, images_targets,
                K=int(getattr(self.cfg.TRAINER.BiMC, 'PGA_K', 8))
            )
        else:
            pga_basis, pga_centers = None, None

        if cls_begin_index != 0:
            if calibrate_novel_vision_proto:
                print(f'calibrate vision proto on class [{class_index}]')
                images_proto = self.soft_calibration(self.base_vision_prototype, images_proto)
        else:
            self.base_vision_prototype = images_proto


        cov_images = torch.cov(images_features.T)

        if cls_begin_index == 0:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_BASE) 
        else:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_INC)

        
        print('finish loading covariance')

        return {
            'description_proto': description_proto,
            'image_description_counts': image_description_counts,  # [C_task] or None
            'description_features': description_features,
            'description_targets': description_targets,

            'text_features': text_features,
            'text_targets': text_targets,           
  
            'image_proto': images_proto,        # [C_task, D]
            'pga_basis': pga_basis,             # [C_task, D, K] or None
            'pga_centers': pga_centers,         # [C_task, D] or None
            'images_features': images_features,
            'images_targets': images_targets,
            'cov_image': cov_images,
            
            'class_index': class_index,
            'sample_cnt': len(images_features)
        }

   

    def forward_ours(self, images, num_cls, num_base_cls,
                           image_proto, cov_image,
                           description_proto,
                           description_features, description_targets,
                           text_features,
                           beta, pga_basis=None, pga_centers=None,
                           return_pga_scores=False):
    
        def knn_similarity_scores(queries, support_features, support_labels):
            """
            Compute the similarity between each query sample and all support samples,
            and retrieve the maximum score for each class per query.
            """
            # Ensure all inputs are on the same device
            device = queries.device
            support_features = support_features.to(device)
            support_labels = support_labels.to(device)
            similarity_scores = torch.matmul(queries, support_features.T)
            k = torch.max(support_labels) + 1
            max_scores = torch.full((queries.size(0), k), float('-inf'), device=device)
            expanded_labels = support_labels.unsqueeze(0).expand(queries.size(0), -1)
            for label in range(k):
                label_mask = (expanded_labels == label)
                masked_scores = similarity_scores.masked_fill(~label_mask, float('-inf'))
                max_scores[:, label] = torch.max(masked_scores, dim=1).values
            return max_scores


        def _mahalanobis(dist, cov_inv):
            """
            Compute the Mahalanobis distance between feature vectors and a class prototype.
            """
            left_term = torch.matmul(dist, cov_inv)
            mahal = torch.matmul(left_term, dist.T)
            return torch.diag(mahal)


        def _cov_forward(feat, proto, cov):
            """
            Perform a forward pass computing negative Mahalanobis distance between 
            features and each class prototype using a shared covariance matrix.
            """
            maha_dist = []
            inv_covmat = torch.pinverse(cov.to(dtype=torch.float32))
            inv_covmat = inv_covmat.to(dtype=proto.dtype)
            for cl in range(num_cls):
                distance = feat - proto[cl]
                dist = _mahalanobis(distance, inv_covmat)
                maha_dist.append(dist)
            maha_dist = torch.stack(maha_dist)
            logits = -maha_dist.T
            return logits
        

        # Normalize the image features
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)

        if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T
        else:
            lambda_t = 0.0

        # Here we compute the classifier after modality calibration. 
        # Note that image_proto has already been calibrated in the `build_task_statistics` function.
        fused_proto = beta * ((1 - lambda_t) * text_features + lambda_t * description_proto) + (1 - beta) * image_proto        
        fused_proto = F.normalize(fused_proto, dim=-1)  
        logits_proto_fused = img_feat @ fused_proto.t()  # [B, C]
        pga_scores = None  # Optional [B, C] scores before PGA_WEIGHT.
        if pga_basis is not None and pga_centers is not None:
            # img_feat [B,D], pga_basis [C,D,K], pga_centers [C,D]
            # pga_scores [B,C]: negative tangent-subspace reconstruction error.
            pga_scores = self.attribute_logits(img_feat, pga_basis, pga_centers)
            pga_weight = float(getattr(self.cfg.TRAINER.BiMC, 'PGA_WEIGHT', 0.1))
            logits_proto_fused = logits_proto_fused + pga_weight * pga_scores
        prob_fused_proto = F.softmax(logits_proto_fused, dim=-1)  # [B, C]

        logits_cov = _cov_forward(img_feat, image_proto, cov_image)
        logits_knn = knn_similarity_scores(img_feat, description_features, description_targets)    
        prob_cov = F.softmax(logits_cov / 512, dim=-1)
        prob_knn = F.softmax(logits_knn, dim=-1)

        NUM_BASE_CLS = num_base_cls
        use_diversity = self.cfg.TRAINER.BiMC.USING_ENSEMBLE
        if use_diversity:
            ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        else:
            ensemble_alpha = 1.0

        base_probs = ensemble_alpha * prob_fused_proto[:, :NUM_BASE_CLS] + (1 - ensemble_alpha) * prob_cov[:, :NUM_BASE_CLS]
        inc_probs = ensemble_alpha * prob_fused_proto[:, NUM_BASE_CLS:] + (1 - ensemble_alpha) * prob_knn[:, NUM_BASE_CLS:]

        prob_fused = torch.cat([base_probs, inc_probs], dim=1)
        logits = prob_fused  # [B, C]
        if return_pga_scores:
            return logits, pga_scores  # [B, C], [B, C] or None
        return logits



    @torch.no_grad()
    def extract_img_feature(self, images):
        images = images.to(self.device)
        image_features = self.clip_model.encode_image(images)
        return image_features


    @torch.no_grad()
    def forward(self, images):
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)
        classifier = F.normalize(self.classifier_weights, dim=-1)
        logits = 100. * img_feat @ classifier.t()
        return logits



    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets