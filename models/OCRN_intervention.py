from typing import final
import torch
import torch.nn as nn
import math

from models.base_models import OcrnBaseModel, MLP, ParallelMLP, Aggregator, build_counterfactual, CounterfactualHingeLoss


@final
class FullSelfAttention(nn.Module):
    def __init__(self, feat_dim, cond_dim, hidden_dim, args):
        """ output = f(input, condition)
        in_dim/cond_dim/out_dim = dimension of input/condition/output
        fc_in_hid/fc_cond_hid = hidden layers of fc after input/condition
        fc_out_hid = hidden layers of fc before output
        """
        super(FullSelfAttention, self).__init__()

        fc_in_hid   = args.fc_pre
        fc_cond_hid = args.fc_att
        fc_out_hid  = args.fc_compress

        self.fc_feat_Q = MLP(feat_dim, hidden_dim, fc_in_hid, args.batchnorm, bias=False)
        self.fc_feat_V = MLP(feat_dim, hidden_dim, fc_in_hid, args.batchnorm, bias=False)
        self.fc_feat_K = MLP(feat_dim, hidden_dim, fc_in_hid, args.batchnorm, bias=False)
        self.fc_cond_Q = MLP(cond_dim, hidden_dim, fc_cond_hid, args.batchnorm, bias=False)
        self.fc_cond_V = MLP(cond_dim, hidden_dim, fc_cond_hid, args.batchnorm, bias=False)
        self.fc_cond_K = MLP(cond_dim, hidden_dim, fc_cond_hid, args.batchnorm, bias=False)

        self.rtemp = 1.0/math.sqrt(hidden_dim)

        self.fc_out = MLP(2*hidden_dim, feat_dim, fc_out_hid, args.batchnorm, out_relu=args.out_relu)


    def forward(self, feat, cond, in_postproc=lambda x:x, cond_postproc=lambda x:x):
        feat_Q = in_postproc( self.fc_feat_Q(feat) ) # (bz*obj, hid_dim)
        feat_V = in_postproc( self.fc_feat_V(feat) )
        feat_K = in_postproc( self.fc_feat_K(feat) )

        cond_Q = cond_postproc( self.fc_cond_Q(cond) )
        cond_V = cond_postproc( self.fc_cond_V(cond) )
        cond_K = cond_postproc( self.fc_cond_K(cond) )

        K_diff = (feat_K - cond_K) * self.rtemp

        KQ_ff_fc = (feat_Q * K_diff).sum(-1) # (bz*obj, )
        KQ_cf_cc = (cond_Q * K_diff).sum(-1)

        feat_att_f = torch.sigmoid(KQ_ff_fc).unsqueeze(-1)
        cond_att_f = torch.sigmoid(KQ_cf_cc).unsqueeze(-1)

        V_diff = (feat_V - cond_V)
        hid_feat = V_diff*feat_att_f + cond_V
        hid_cond = V_diff*cond_att_f + cond_V
        hidden = torch.cat([hid_feat, hid_cond], -1)
        out = self.fc_out(hidden)
        
        return out




# @final
class Model(OcrnBaseModel):
    def __init__(self, dataset, args):
        super(Model, self).__init__(dataset, args)

        # model param
        
        self.fc_feat2attr = MLP(self.feat_dim, args.attr_rep_dim, args.fc_feat2attr, args.batchnorm, out_relu=args.out_relu, out_bn=args.batchnorm)
        self.fc_feat2aff = MLP(self.feat_dim + args.attr_rep_dim, args.aff_rep_dim, args.fc_feat2aff, args.batchnorm, out_relu=args.out_relu, out_bn=args.batchnorm)

        self.attr_instantialize = FullSelfAttention(args.attr_rep_dim, self.feat_dim, args.attr_hidden_rep_dim, args=args)
        self.aff_instantialize = FullSelfAttention(args.aff_rep_dim, self.feat_dim + args.aggr_rep_dim, args.aff_hidden_rep_dim, args=args)

        
        self.aggregator = Aggregator(self.args.aggregation, args, self.num_attr)

        self.parallel_attr_feat = ParallelMLP(
            args.attr_out_rep_dim, args.parallel_attr_rep_dim, num_para=self.num_attr,
            hidden_layers=args.fc_para_feat, layernorm=args.layernorm, out_relu=args.out_relu)

        self.attr_auxIA_classifier = ParallelMLP(args.parallel_attr_rep_dim, 1, num_para=self.num_attr, hidden_layers=args.fc_cls, 
            layernorm=args.layernorm, share_last_fc=True)

        self.attr_IA_classifier = MLP(args.attr_rep_dim, self.num_attr, hidden_layers=args.fc_cls, batchnorm=args.batchnorm)
        self.aff_IA_classifier = MLP(args.aff_rep_dim, self.num_aff, hidden_layers=args.fc_cls, batchnorm=args.batchnorm)
        assert args.sep_CA_cls
        self.attr_CA_classifier = MLP(args.attr_rep_dim, self.num_attr, hidden_layers=args.fc_cls, batchnorm=args.batchnorm)
        self.aff_CA_classifier = MLP(args.aff_rep_dim, self.num_aff, hidden_layers=args.fc_cls, batchnorm=args.batchnorm)


        self.mseloss = torch.nn.MSELoss()
        self.hinge = CounterfactualHingeLoss(args.counterfactual_margin)




    def forward(self, batch, require_loss=True):
        if self.backbone:
            feature = self.backbone(batch["image"], batch["main_bbox"])
            batch["gt_attr"] = torch.cat(batch["gt_attr"], 0)
            batch["gt_aff"] = torch.cat(batch["gt_aff"], 0)
        else:
            feature = batch["image"]
        
        batchsize = feature.size(0)

        gt_all_CAttr_vec = self.category_attr
        gt_all_CAff_vec = self.category_aff

        #  Attibute module

        feat_CAttr = self.fc_feat2attr(self.mean_obj_features) # (n_obj, dim_attr)
        feat_IAttr = self.attr_instantialize(
            feat_CAttr, feature,
            in_postproc = lambda x:x.unsqueeze(0).expand(batchsize, -1, -1),
            cond_postproc = lambda x:x.unsqueeze(1).expand(-1, self.num_obj, -1)
        ) # (n_obj, dim), (bz, dim)  -> (bz, n_obj, dim)

        # feat_IAttr = self.attr_inst_bn(feat_IAttr)
        feat_mean_IAttr = torch.einsum("ijk,j->ik", feat_IAttr, self.obj_frequence)
        # (bz, dim_attr)

        logit_CAttr = self.attr_CA_classifier(feat_CAttr)
        logit_IAttr = self.attr_IA_classifier(feat_mean_IAttr)

        feat_parallel_IAttr = self.parallel_attr_feat(feat_mean_IAttr.unsqueeze(1).expand(-1,self.num_attr, -1))
        logit_aux_IAttr = self.attr_auxIA_classifier(feat_parallel_IAttr).squeeze(-1)


        #  Affordance module

        feat_aggr_IAttr = self.aggregator(feat_parallel_IAttr)

        feat_CAff = self.fc_feat2aff(
            torch.cat([self.mean_obj_features, feat_CAttr], 1)
        ) # (n_obj, dim_aff)

        feat_IAff = self.aff_instantialize(
            feat_CAff, torch.cat([feature, feat_aggr_IAttr], 1),
            in_postproc = lambda x:x.unsqueeze(0).expand(batchsize, -1, -1),
            cond_postproc = lambda x:x.unsqueeze(1).expand(-1, self.num_obj, -1)
        )
        # (n_obj, dim), (bz, dim) -> (bz, n_obj, dim)

        # feat_IAff = self.aff_inst_bn(feat_IAff)
        feat_mean_IAff =  torch.einsum("ijk,j->ik", feat_IAff, self.obj_frequence)
        # (bz, dim_aff)

        logit_CAff = self.aff_CA_classifier(feat_CAff)
        logit_IAff = self.aff_IA_classifier(feat_mean_IAff)

        prob_IAttr = torch.sigmoid(logit_IAttr)
        prob_IAff = torch.sigmoid(logit_IAff)


        if require_loss:
            losses = {}
            
            if self.args.lambda_attr > 0:
                if self.args.lambda_cls_CA>0:
                    losses["loss_attr/CA_cls"] = self.attr_bce(logit_CAttr, gt_all_CAttr_vec)
                if self.args.lambda_cls_IA>0:
                    losses["loss_attr/IA_cls"] = self.attr_bce(logit_IAttr, batch["gt_attr"])
                if self.args.lambda_cls_inst_IA>0:
                    logit_inst_IAttr = self.attr_IA_classifier(feat_IAttr)
                    losses["loss_attr/inst_IA_cls"] = self.attr_bce(
                        logit_inst_IAttr, batch["gt_attr"].unsqueeze(1).expand(-1, self.num_obj, -1))


                if any([x.startswith("loss_attr") for x in losses]):
                    losses["loss_attr/total"] = (
                        self.args.lambda_cls_CA * losses.get("loss_attr/CA_cls", 0.) +
                        self.args.lambda_cls_IA * losses.get("loss_attr/IA_cls", 0.) +
                        self.args.lambda_cls_inst_IA * losses.get("loss_attr/inst_IA_cls", 0.) )

            if self.args.lambda_aff > 0:
                if self.args.lambda_cls_CA>0:
                    losses["loss_aff/CA_cls"] = self.aff_bce(logit_CAff, gt_all_CAff_vec)
                if self.args.lambda_cls_IA>0:
                    losses["loss_aff/IA_cls"] = self.aff_bce(logit_IAff, batch["gt_aff"])
                if self.args.lambda_cls_inst_IA>0:
                    logit_inst_IAff = self.aff_IA_classifier(feat_IAff)
                    losses["loss_aff/inst_IA_cls"] = self.aff_bce( 
                        logit_inst_IAff, batch["gt_aff"].unsqueeze(1).expand(-1, self.num_obj, -1))


                if any([x.startswith("loss_aff") for x in losses]):
                    losses["loss_aff/total"] = (
                        self.args.lambda_cls_CA * losses.get("loss_aff/CA_cls", 0.) +
                        self.args.lambda_cls_IA * losses.get("loss_aff/IA_cls", 0.) +
                        self.args.lambda_cls_inst_IA * losses.get("loss_aff/inst_IA_cls", 0.) )

                
            
            if self.args.lambda_cf > 0 and batch['gt_causal'].shape[0] > 0:
                cf_inst_id, cf_attr_mask, cf_aff_mask = build_counterfactual(
                    batch['gt_causal'], self.num_attr, self.num_aff)

                cf_feat_aggr_IAttr = self.aggregator(feat_parallel_IAttr[cf_inst_id], 1-cf_attr_mask)
                cf_feat_IAff = self.aff_instantialize(
                    feat_CAff, torch.cat([feature[cf_inst_id], cf_feat_aggr_IAttr], 1),
                    in_postproc = lambda x:x.unsqueeze(0).expand(cf_feat_aggr_IAttr.size(0), -1, -1),
                    cond_postproc = lambda x:x.unsqueeze(1).expand(-1, self.num_obj, -1)
                )
                cf_feat_mean_IAff =  torch.einsum("ijk,j->ik", cf_feat_IAff, self.obj_frequence)
                cf_logit_IAff = self.aff_IA_classifier(cf_feat_mean_IAff)
                cf_prob_IAff = torch.sigmoid(cf_logit_IAff)

                loss_cf = self.hinge(
                    cf_prob_IAff, prob_IAff[cf_inst_id], batch['gt_aff'][cf_inst_id], cf_aff_mask
                )
                losses['loss_cf'] = loss_cf


            
            if self.args.lambda_aux_cls>0:
                losses["loss_aux_IA_cls"] = self.attr_bce(logit_aux_IAttr, batch["gt_attr"])
                # NOTE: freeze attr will not freeze this

            
            losses["loss_total"] = (
                self.args.lambda_attr * losses.get("loss_attr/total", 0.) + 
                self.args.lambda_aff * losses.get("loss_aff/total", 0.) + 
                self.args.lambda_cf * losses.get("loss_cf", 0.) + 
                self.args.lambda_aux_cls * losses.get("loss_aux_IA_cls", 0.) )


            return losses
        else:

            preds = {
                "attr_main": prob_IAttr,
                "aff_main": prob_IAff,
            }
            return preds





class TDEModel(Model):
    def forward(self, batch, require_loss=False):
        if self.backbone:
            feature = self.backbone(batch["image"], batch["main_bbox"])
            batch["gt_attr"] = torch.cat(batch["gt_attr"], 0)
            batch["gt_aff"] = torch.cat(batch["gt_aff"], 0)
        else:
            feature = batch["image"]
        
        batchsize = feature.size(0)


        #  Attibute module

        feat_CAttr = self.fc_feat2attr(self.mean_obj_features) # (n_obj, dim_attr)

        feat_IAttr = self.attr_instantialize(
            feat_CAttr, feature,
            in_postproc = lambda x:x.unsqueeze(0).expand(batchsize, -1, -1),
            cond_postproc = lambda x:x.unsqueeze(1).expand(-1, self.num_obj, -1)
        )
        # (n_obj, dim), (bz, dim)  -> (bz, n_obj, dim)
        
        feat_mean_IAttr = torch.einsum("ijk,j->ik", feat_IAttr, self.obj_frequence)
        # (bz, n_attr, dim_attr)

        logit_IAttr = self.attr_IA_classifier(feat_mean_IAttr)

        feat_parallel_IAttr = self.parallel_attr_feat(feat_mean_IAttr.unsqueeze(1).expand(-1,self.num_attr, -1))

        #  Affordance module

        feat_aggr_IAttr = self.aggregator(feat_parallel_IAttr)

        feat_CAff = self.fc_feat2aff(
            torch.cat([self.mean_obj_features, feat_CAttr], 1)
        ) # (n_obj, dim_aff)


        feat_IAff = self.aff_instantialize(
            feat_CAff, torch.cat([feature, feat_aggr_IAttr], 1),
            in_postproc = lambda x:x.unsqueeze(0).expand(batchsize, -1, -1),
            cond_postproc = lambda x:x.unsqueeze(1).expand(-1, self.num_obj, -1)
        )

        feat_mean_IAff =  torch.einsum("ijk,j->ik", feat_IAff, self.obj_frequence)
        # (bz, dim_aff)
        
        logit_IAff = self.aff_IA_classifier(feat_mean_IAff)


        preds = {
            "attr_main": torch.sigmoid(logit_IAttr),
            "aff_main": torch.sigmoid(logit_IAff),
        }
            
        tde = []
        mask_gen = (1.-torch.eye(self.num_attr)).to(feature.device).bool()
        for i in range(self.num_attr):
            masked_feat_aggr_IAttr = self.aggregator(feat_parallel_IAttr, mask_gen[i],
                                                     mask_method=self.args.get("mask_method", "zero"))

            masked_feat_IAff = self.aff_instantialize(
                feat_CAff, torch.cat([feature, masked_feat_aggr_IAttr], 1),
                in_postproc = lambda x:x.unsqueeze(0).expand(batchsize, -1, -1),
                cond_postproc = lambda x:x.unsqueeze(1).expand(-1, self.num_obj, -1) )
            # (n_obj, dim), (bz, dim) -> (bz*n_obj, dim)

            masked_feat_mean_IAff =  torch.einsum("ijk,j->ik", masked_feat_IAff, self.obj_frequence)
            counterfactual_logit_IAff = self.aff_IA_classifier(masked_feat_mean_IAff)
            counterfactual_prob_IAff = torch.sigmoid(counterfactual_logit_IAff)
            tde.append(preds["aff_main"] - counterfactual_prob_IAff)
        tde = torch.stack(tde, dim=1)

        preds.update({
            'tde_main': tde
        })

        return preds
