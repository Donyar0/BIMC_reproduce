import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
from models.vdt_self_attention import SelfAttentionTextAdapter, pack_description_bank, mean_description_prototypes

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
        # CLIP-A-self is trained on base images only, then frozen forever.
        dim = int(self.clip_model.text_projection.shape[1])
        self.text_adapter = SelfAttentionTextAdapter(
            embed_dim=dim,
            reduction=cfg.TRAINER.BiMC.VDT_REDUCTION,
            ratio=cfg.TRAINER.BiMC.VDT_RATIO,
            dropout=cfg.TRAINER.BiMC.VDT_DROPOUT,
        ).to(self.device)
        self.text_adapter_trained = False


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
    def inference_vdt_features(self, class_names, gpt_path, cls_begin_index):
        """Encode name-conditioned descriptions for the CLIP-A-self adapter.

        The original BiMC standalone description features are retained for KNN.
        A VDT prompt consists of a class prompt followed by a description.
        """
        with open(gpt_path, "r") as handle:
            descriptions_by_name = json.load(handle)
        descriptions_by_name = {
            str(k).lower().replace("_", " ").replace("-", " "): v
            for k, v in descriptions_by_name.items()
        }
        all_features, all_labels = [], []
        for offset, name in enumerate(class_names):
            normalized_name = str(name).lower().replace("_", " ").replace("-", " ")
            descriptions = descriptions_by_name[normalized_name]
            if not descriptions:
                raise ValueError("No descriptions available for class " + str(name))
            class_prompt = self.template[0].format(normalized_name)
            # Each description is already a complete sentence in BiMC's JSON.
            # The prompt uses text-level concatenation, not feature concatenation.
            prompts = [class_prompt + " " + str(desc) for desc in descriptions]
            parts = []
            for start in range(0, len(prompts), 64):
                tokens = clip.tokenize(prompts[start:start + 64], truncate=True).to(self.device)
                encoded = self.clip_model.encode_text(tokens)
                parts.append(F.normalize(encoded, dim=-1))
            feature = torch.cat(parts, dim=0)
            all_features.append(feature)
            all_labels.append(torch.full(
                (feature.size(0),), cls_begin_index + offset,
                dtype=torch.long, device=feature.device
            ))
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def vdt_prototypes(self, features, labels):
        packed, mask, classes = pack_description_bank(features, labels)
        if self.cfg.TRAINER.BiMC.VDT_MODE == "mean":
            prototypes = mean_description_prototypes(packed, mask)
        elif self.cfg.TRAINER.BiMC.VDT_MODE == "attention":
            prototypes = self.text_adapter(packed, mask)
        else:
            raise ValueError("VDT_MODE must be 'mean' or 'attention'")
        return prototypes, classes

    def train_base_text_adapter(self, base_state):
        """Train solely on base-session image features and their labels.

        CLIP image/text encoders are frozen. Only attention parameters update.
        Does not access any incremental class training images or descriptions.
        """
        if self.cfg.TRAINER.BiMC.VDT_MODE in ("mean", "original"):
            self.text_adapter.eval()
            self.text_adapter_trained = False
            print("VDT mode={}: skip Self-Attention training".format(
                self.cfg.TRAINER.BiMC.VDT_MODE
            ))
            return
        self.text_adapter.train()
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            self.text_adapter.parameters(),
            lr=self.cfg.TRAINER.BiMC.VDT_LR,
            weight_decay=self.cfg.TRAINER.BiMC.VDT_WEIGHT_DECAY
        )
        image_features = F.normalize(base_state['images_features'].detach().float(), dim=-1)
        targets = base_state['images_targets'].detach().long()
        packed, mask, classes = pack_description_bank(
            base_state['vdt_features'].detach(), base_state['vdt_targets']
        )
        if not torch.equal(classes, torch.arange(len(classes), device=classes.device)):
            raise ValueError("Base classes must have contiguous zero-based labels")
        num_images = image_features.size(0)
        batch_size = int(self.cfg.TRAINER.BiMC.VDT_BATCH_SIZE)
        if batch_size < 1:
            raise ValueError("VDT_BATCH_SIZE must be positive")
        with torch.enable_grad():
            for epoch in range(int(self.cfg.TRAINER.BiMC.VDT_EPOCHS)):
                order = torch.randperm(num_images, device=image_features.device)
                epoch_loss = 0.0
                for start in range(0, num_images, batch_size):
                    idx = order[start:start + batch_size]
                    optimizer.zero_grad()
                    class_weights = self.text_adapter(packed, mask)
                    logits = 100.0 * image_features[idx] @ class_weights.t()
                    loss = F.cross_entropy(logits, targets[idx])
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item() * idx.numel()
                print("VDT-Adapter base epoch={}/{} CE={:.6f}".format(
                    epoch + 1, self.cfg.TRAINER.BiMC.VDT_EPOCHS,
                    epoch_loss / num_images
                ))
        self.text_adapter.eval()
        for param in self.text_adapter.parameters():
            param.requires_grad_(False)
        self.text_adapter_trained = True

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

        description_features, description_targets, description_proto = \
                                  self.inference_all_description_feature(class_names=class_names, 
                                  gpt_path=self.cfg.DATASET.GPT_PATH,
                                  cls_begin_index=cls_begin_index)
        
        if self.cfg.TRAINER.BiMC.VDT_MODE == "original":
            # Avoid extra VDT encoding for the genuine original-BiMC control.
            vdt_features = description_features.new_empty((0, description_features.size(-1)))
            vdt_targets = description_targets.new_empty((0,))
        else:
            vdt_features, vdt_targets = self.inference_vdt_features(
                class_names, self.cfg.DATASET.GPT_PATH, cls_begin_index
            )

        images_features, images_targets, images_proto = \
                                    self.inference_all_img_feature(loader, cls_begin_index)

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
            'vdt_features': vdt_features,
            'vdt_targets': vdt_targets,
            'description_features': description_features,
            'description_targets': description_targets,

            'text_features': text_features,
            'text_targets': text_targets,           
  
            'image_proto': images_proto,
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
                           beta):
    
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

        # CLIP-A-self text prototype is computed by Runner and supplied here.
        # Legacy BiMC text interpolation has been replaced entirely.
        if self.cfg.TRAINER.BiMC.VDT_MODE == "original":
            # Exact original BiMC text calibration as an in-branch control.
            original_t = self.cfg.TRAINER.BiMC.VDT_ORIGINAL_LAMBDA_T
            calibrated_text = (1 - original_t) * text_features + original_t * description_proto
        else:
            calibrated_text = text_features
        fused_proto = beta * calibrated_text.to(dtype=image_proto.dtype) + (1 - beta) * image_proto        
        fused_proto = F.normalize(fused_proto, dim=-1)  
        logits_proto_fused = img_feat @ fused_proto.t()
        prob_fused_proto = F.softmax(logits_proto_fused, dim=-1)

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
        logits = prob_fused
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