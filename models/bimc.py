import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
from models.vdt_linear import LinearTextAdapter, pack_description_bank
from models.vdt_objective import linear_adapter_ce, support_feature_moments
from models.vdt_visual import (VisionSelfAttentionAdapter, train_visual_aggregator,
                               aggregate_visual_classes)

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
        # Base-session supervised text calibration; optional per-session 5-shot updates.
        mode = cfg.TRAINER.BiMC.VDT_MODE
        if mode not in ("linear", "original"):
            raise ValueError("VDT_MODE must be 'linear' or 'original'")
        self.text_adapter = None
        self.text_adapter_trained = False
        if mode == "linear":
            dim = int(self.clip_model.text_projection.shape[1])
            self.text_adapter = LinearTextAdapter(
                embed_dim=dim, ratio=cfg.TRAINER.BiMC.VDT_RATIO,
                dropout=cfg.TRAINER.BiMC.VDT_DROPOUT
            ).to(self.device)
            print("VDT_MODE=linear | trainable adapter parameters={}".format(
                sum(p.numel() for p in self.text_adapter.parameters())
            ))
        else:
            print("VDT_MODE=original | trainable adapter parameters=0")

        self.visual_adapter = None
        self.visual_adapter_trained = False
        if cfg.TRAINER.BiMC.VDT_USE_VISUAL_ADAPTER:
            dim = int(self.clip_model.visual.output_dim)
            # Independent, reproducible initialization: text-only vs dual
            # comparisons must not change Linear Adapter initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(cfg.SEED) + 91731)
                visual_net = VisionSelfAttentionAdapter(
                    embed_dim=dim,
                    reduction=cfg.TRAINER.BiMC.VDT_VISUAL_REDUCTION,
                    ratio=cfg.TRAINER.BiMC.VDT_VISUAL_RATIO,
                    dropout=cfg.TRAINER.BiMC.VDT_VISUAL_DROPOUT,
                )
            self.visual_adapter = visual_net.to(self.device)
            print("VDT_VISUAL_ATTENTION | trainable parameters={}".format(
                sum(p.numel() for p in self.visual_adapter.parameters())
            ))


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
        if self.text_adapter is None:
            raise RuntimeError("Text calibration is only available in linear mode")
        packed, mask, classes = pack_description_bank(features, labels)
        return self.text_adapter(packed, mask), classes

    def train_base_visual_adapter(self, base_state):
        """Train on base TRAIN images only, then freeze until all sessions end."""
        if self.visual_adapter is None:
            return
        if self.visual_adapter_trained:
            raise RuntimeError("Visual Adapter must be trained only once")
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        features = base_state["images_features"].detach().float()
        labels = base_state["images_targets"].detach().long()
        class_ids = torch.as_tensor(base_state["class_index"],device=labels.device)
        if not torch.equal(class_ids,torch.arange(len(class_ids),device=labels.device)):
            raise ValueError("Base class IDs must be contiguous and zero-based")
        history = train_visual_aggregator(
            self.visual_adapter, features, labels,
            num_classes=len(class_ids),
            epochs=int(self.cfg.TRAINER.BiMC.VDT_VISUAL_EPOCHS),
            steps_per_epoch=int(self.cfg.TRAINER.BiMC.VDT_VISUAL_STEPS),
            support_per_class=int(self.cfg.TRAINER.BiMC.VDT_VISUAL_EPISODE_SHOTS),
            query_per_class=int(self.cfg.TRAINER.BiMC.VDT_VISUAL_EPISODE_QUERIES),
            lr=float(self.cfg.TRAINER.BiMC.VDT_VISUAL_LR),
            weight_decay=float(self.cfg.TRAINER.BiMC.VDT_WEIGHT_DECAY),
            seed=int(self.cfg.SEED),
        )
        self.visual_adapter_trained = True
        print("VDT_VISUAL_TRAIN_JSON:" + json.dumps({
            "training":"base_only_disjoint_episodic_support_query",
            "num_base_images":int(len(features)),
            "epochs":len(history),
            "steps_per_epoch":int(self.cfg.TRAINER.BiMC.VDT_VISUAL_STEPS),
            "final_episodic_ce":history[-1],
        },sort_keys=True))

    @torch.no_grad()
    def vdt_visual_prototypes(self, state, task_id):
        """Generate current-session prototypes with a frozen visual attention."""
        if self.visual_adapter is None or not self.visual_adapter_trained:
            raise RuntimeError("Visual Adapter must be base-trained before aggregation")
        if self.visual_adapter.training or any(p.requires_grad for p in self.visual_adapter.parameters()):
            raise RuntimeError("Visual Adapter must remain frozen during inference")
        class_ids = torch.as_tensor(state["class_index"],
                                    device=state["images_targets"].device)
        result = aggregate_visual_classes(
            self.visual_adapter,
            state["images_features"].detach(),
            state["images_targets"].detach(),
            class_ids,
            max_support=int(self.cfg.TRAINER.BiMC.VDT_VISUAL_MAX_SUPPORT),
        )
        if task_id == 0:
            self.base_vision_prototype = result.detach().clone()
        elif self.cfg.TRAINER.BiMC.VISION_CALIBRATION:
            result = self.soft_calibration(self.base_vision_prototype, result)
        # CLIP fp16 inference expects image prototypes in its image-feature
        # dtype; keep the base calibration reference in fp32 internally.
        return result.to(dtype=state["image_proto"].dtype)

    def train_base_text_adapter(self, base_state):
        """Train linear adapter on base-session images only; freeze in FSCIL."""
        if self.text_adapter is None:
            print("VDT_MODE=original: no text adapter to train")
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
        objective = str(self.cfg.TRAINER.BiMC.VDT_TRAIN_OBJECTIVE)
        if objective == "fusion_ce" and self.visual_adapter is not None:
            raise ValueError(
                "fusion_ce + visual attention requires an attention-aware "
                "leave-one-out positive prototype; run the three-arm "
                "comparison with --objective text_ce"
            )
        beta = float(self.cfg.DATASET.BETA)
        if objective == "fusion_ce":
            visual = base_state["image_proto"].detach().float()
            support_sums, support_counts = support_feature_moments(
                image_features, targets, len(classes)
            )
        else:
            visual, support_sums, support_counts = None, None, None
        print("VDT_BASE_OBJECTIVE_JSON:" + json.dumps({
            "objective": objective, "beta": beta,
            "leave_one_out_positive": objective == "fusion_ce",
        }, sort_keys=True))
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
                    loss = linear_adapter_ce(
                        image_features[idx], targets[idx], class_weights,
                        visual_prototypes=visual, beta=beta, mode=objective,
                        support_sums=support_sums, support_counts=support_counts,
                    )
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

    def train_incremental_text_adapter(self, current_state, merged_state, task_id):
        """Optional session-wise CE fine-tuning on NEW training shots only.

        The model is base-trained first. At session t>0, all seen class text
        descriptions act as competing classifier weights, but only the current
        session's 5-shot image features/labels provide CE supervision.
        No replay, distillation, test images, or extra losses are used.
        Trainable Linear Adapter is refrozen before evaluation.
        """
        epochs = int(self.cfg.TRAINER.BiMC.VDT_INC_EPOCHS)
        if epochs < 0:
            raise ValueError("Incremental epochs cannot be negative")
        if self.text_adapter is None or task_id == 0 or epochs == 0:
            return
        if not self.text_adapter_trained:
            raise RuntimeError("Train base adapter before incremental fine-tuning")
        lr = float(self.cfg.TRAINER.BiMC.VDT_INC_LR)
        if lr <= 0:
            raise ValueError("Incremental learning rate must be positive")
        packed, mask, classes = pack_description_bank(
            merged_state["vdt_features"].detach(),
            merged_state["vdt_targets"].detach(),
        )
        expected = torch.arange(len(classes), device=classes.device)
        if not torch.equal(classes, expected):
            raise ValueError("Seen classes must be contiguous, zero-based IDs")
        images = F.normalize(current_state["images_features"].detach().float(), dim=-1)
        labels = current_state["images_targets"].detach().long()
        current_ids = torch.as_tensor(current_state["class_index"], device=labels.device)
        if images.numel() == 0 or not (labels[:, None] == current_ids[None, :]).any(dim=1).all():
            raise ValueError("Incremental training images must be from the new session")
        if labels.min() < 0 or labels.max() >= len(classes):
            raise ValueError("Incremental labels do not match seen class classifier")

        objective = str(self.cfg.TRAINER.BiMC.VDT_INC_OBJECTIVE)
        beta = float(self.cfg.DATASET.BETA)
        if objective == "fusion_ce":
            visual = merged_state["image_proto"].detach().float()
            # Old class support features are NOT used to fine-tune the
            # adapter. Only new-session supports build LOO moments.
            support_sums, support_counts = support_feature_moments(
                images, labels, len(classes)
            )
            # Reproduce BiMC's novel-class prototype calibration after
            # removing the current query from its support centroid.
            def loo_calibration(loo_features):
                if not self.cfg.TRAINER.BiMC.VISION_CALIBRATION:
                    return loo_features
                return self.soft_calibration(self.base_vision_prototype, loo_features)
        else:
            visual, support_sums, support_counts = None, None, None
            loo_calibration = None

        adapter = self.text_adapter
        # Measure text-prototype drift caused by new-shot-only updating.
        # Evaluation prototypes are always produced with dropout disabled.
        adapter.eval()
        with torch.no_grad():
            prototype_before = adapter(packed, mask).detach().clone()
        params_before = [p.detach().clone() for p in adapter.parameters()]
        for p in adapter.parameters():
            p.requires_grad_(True)
        adapter.train()
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=lr,
            weight_decay=self.cfg.TRAINER.BiMC.VDT_WEIGHT_DECAY,
        )
        batch_size = int(self.cfg.TRAINER.BiMC.VDT_BATCH_SIZE)
        if batch_size <= 0:
            raise ValueError("VDT_BATCH_SIZE must be positive")
        try:
            with torch.enable_grad():
                for epoch in range(epochs):
                    order = torch.randperm(images.size(0), device=images.device)
                    loss_total = 0.0
                    for start in range(0, images.size(0), batch_size):
                        idx = order[start:start + batch_size]
                        optimizer.zero_grad()
                        classifier = adapter(packed, mask)
                        loss = linear_adapter_ce(
                            images[idx], labels[idx], classifier,
                            visual_prototypes=visual, beta=beta, mode=objective,
                            support_sums=support_sums, support_counts=support_counts,
                            calibrate_loo=loo_calibration,
                        )
                        loss.backward()
                        optimizer.step()
                        loss_total += loss.detach().item() * idx.numel()
                    print("VDT-Adapter incremental task={} epoch={}/{} CE={:.6f}".format(
                        task_id, epoch + 1, epochs, loss_total / images.size(0)
                    ))
        finally:
            adapter.eval()
            for p in adapter.parameters():
                p.requires_grad_(False)
        with torch.no_grad():
            delta_sq = sum(
                (p.detach().float() - old.float()).square().sum().item()
                for p, old in zip(adapter.parameters(), params_before)
            )
        with torch.no_grad():
            prototype_after = adapter(packed, mask).detach()
            prototype_cosines = F.cosine_similarity(
                prototype_before, prototype_after, dim=-1
            )
            previous_ids = ~(
                classes[:, None] == current_ids.to(classes.device)[None, :]
            ).any(dim=1)
            new_ids_mask = ~previous_ids
            previous_cosine_mean = (
                float(prototype_cosines[previous_ids].mean().item())
                if previous_ids.any() else None
            )
            new_cosine_mean = (
                float(prototype_cosines[new_ids_mask].mean().item())
                if new_ids_mask.any() else None
            )
        print("VDT_INCREMENTAL_UPDATE_JSON:" + json.dumps({
            "session": task_id,
            "num_train_samples": int(images.size(0)),
            "num_new_classes": int(current_ids.numel()),
            "num_seen_classes": int(len(classes)),
            "epochs": epochs,
            "lr": lr,
            "objective": objective,
            "beta": beta,
            "leave_one_out_positive": objective == "fusion_ce",
            "parameter_delta_l2": delta_sq ** 0.5,
            "previous_text_proto_cosine_before_after": previous_cosine_mean,
            "new_text_proto_cosine_before_after": new_cosine_mean,
        }, sort_keys=True))

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