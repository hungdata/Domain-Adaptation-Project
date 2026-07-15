import re
import os

bases = ["dann", "cdan", "cdane", "dsan", "dsane"]

for base in bases:
    in_path = f"methods/train_{base}.py"
    out_path = f"methods/train_{base}_ir.py"
    
    with open(in_path, "r") as f:
        code = f.read()
        
    # 1. METHOD Name
    method_upper = base.upper()
    code = re.sub(rf'METHOD\s*=\s*"{method_upper}"', f'METHOD = "{method_upper}_IR"', code)
    
    # 2. Fix Imports
    if "DomainDiscriminator" not in code:
        code = code.replace("ClassifierHead,", "ClassifierHead, DomainDiscriminator,")
    if "BOTTLENECK_DIM" not in code:
        code = code.replace("get_device,", "get_device, BOTTLENECK_DIM,")
        
    # 3. Model Init
    if "dsan" in base:
        init_search = re.compile(r'    F_ext\s*=\s*FeatureExtractor\(\)\.to\(device\).*?param_groups = build_param_groups\(F_ext, bottleneck, classifier, lr=args\.lr\)', re.DOTALL)
        init_replace = """    F_ext       = FeatureExtractor().to(device)
    bottleneck  = Bottleneck().to(device)
    classifier  = ClassifierHead().to(device)
    aux_disc    = DomainDiscriminator(in_dim=BOTTLENECK_DIM).to(device)
    modules     = {"F_ext": F_ext, "bottleneck": bottleneck, "classifier": classifier, "aux_disc": aux_disc}

    param_groups = build_param_groups(F_ext, bottleneck, classifier, lr=args.lr)
    param_groups.append({"params": aux_disc.parameters(), "lr": args.lr})"""
        code = init_search.sub(init_replace, code)
    else:
        init_search = re.compile(r'    F_ext\s*=\s*FeatureExtractor\(\)\.to\(device\).*?param_groups = build_param_groups\(F_ext, bottleneck, classifier, lr=args\.lr\)', re.DOTALL)
        init_replace = """    F_ext       = FeatureExtractor().to(device)
    bottleneck  = Bottleneck().to(device)
    classifier  = ClassifierHead().to(device)
    aux_disc    = DomainDiscriminator(in_dim=BOTTLENECK_DIM).to(device)
    domain_disc = DomainDiscriminator(in_dim=BOTTLENECK_DIM * NUM_CLASSES if 'cdan' in METHOD.lower() else BOTTLENECK_DIM).to(device)
    modules     = {"F_ext": F_ext, "bottleneck": bottleneck, "classifier": classifier, "aux_disc": aux_disc, "domain_disc": domain_disc}

    param_groups = build_param_groups(F_ext, bottleneck, classifier, lr=args.lr)
    param_groups.append({"params": aux_disc.parameters(), "lr": args.lr})
    param_groups.append({"params": domain_disc.parameters(), "lr": args.lr})"""
        code = init_search.sub(init_replace, code)

    # 4. Losses init
    code = code.replace("ce_loss_fn   = nn.CrossEntropyLoss()", "ce_loss_fn   = nn.CrossEntropyLoss()\n    ce_none_fn   = nn.CrossEntropyLoss(reduction='none')\n    bce_loss_fn  = nn.BCEWithLogitsLoss()")
    
    # 5. Train modes
    code = code.replace("F_ext.train(); bottleneck.train(); classifier.train()", "F_ext.train(); bottleneck.train(); classifier.train(); aux_disc.train()")
    if "dann" in base or "cdan" in base:
        code = code.replace("aux_disc.train()", "aux_disc.train(); domain_disc.train()")
    
    # 6. Epoch accumulators
    code = code.replace("epoch_cls_loss = 0.0", "epoch_cls_loss = 0.0; epoch_aux_domain_loss = 0.0; epoch_ir_weight = 0.0")
    
    # 7. WARMUP block
    warmup_search = re.compile(r'                cls_loss\s*=\s*ce_loss_fn\(src_logits, src_labels\)\n(.*?)(total_loss\s*=\s*[^\n]+)', re.DOTALL)
    
    def warmup_replace(m):
        return f"                cls_loss   = ce_loss_fn(src_logits, src_labels)\n                aux_domain_loss = torch.zeros((), device=device)\n                ir_weight_mean = 1.0\n{m.group(1)}{m.group(2)}"
    code = warmup_search.sub(warmup_replace, code, count=1)
    
    # 8. DA Phase block
    da_start = "                # DA PHASE - GHÉP BATCH"
    aux_logic = """
                # -------------------------------------------------------------
                # TRUTHFUL AUXILIARY DISCRIMINATOR FOR IMPORTANCE REWEIGHTING
                # -------------------------------------------------------------
                concat_bn_detach = torch.cat([src_bn.detach(), tgt_bn.detach()], dim=0)
                concat_d_aux     = aux_disc(concat_bn_detach)
                src_d_aux, tgt_d_aux = concat_d_aux[:bs_src], concat_d_aux[bs_src:]
                
                d_loss_src_aux   = bce_loss_fn(src_d_aux, torch.ones_like(src_d_aux))
                d_loss_tgt_aux   = bce_loss_fn(tgt_d_aux, torch.zeros_like(tgt_d_aux))
                aux_domain_loss  = 0.5 * (d_loss_src_aux + d_loss_tgt_aux)
                
                src_prob = torch.sigmoid(src_d_aux.detach())
                w_s = 1.0 - src_prob
                w_s = w_s / (w_s.mean() + 1e-5)
                ir_weight_mean = w_s.mean().item()
                
                ce_none = ce_none_fn(src_logits, src_labels)
                cls_loss = (ce_none * w_s.view(-1)).mean()
                # -------------------------------------------------------------
"""
    
    tgt_logits_line = "                tgt_logits = classifier(tgt_bn)"
    # Insert right after tgt_logits = classifier(tgt_bn) OR src_logits = classifier(src_bn)
    if tgt_logits_line in code:
        code = code.replace(tgt_logits_line, tgt_logits_line + "\n" + aux_logic)
    else:
        # Some methods don't explicitly compute tgt_logits in a separate variable if not needed
        pass
        
    # Wait, in the user's report, they said "UnboundLocalError: cannot access local variable 'cls_loss' where it is not associated with a value". 
    # This means the code I generated REMOVED the cls_loss definition but didn't successfully insert the new one!
    # Let's fix the insertion logic.
    
    # The safest way is to replace the FIRST occurrence of cls_loss calculation in DA phase.
    # Let's split by DA start
    da_split = code.split(da_start)
    if len(da_split) == 2:
        da_phase = da_split[1]
        
        # Replace plain cls_loss with aux_logic
        da_phase = re.sub(r'                cls_loss\s*=\s*ce_loss_fn\(src_logits, src_labels\)', aux_logic, da_phase, count=1)
        
        # 9. Total loss update in DA phase
        if "domain_loss" in da_phase and "total_loss = cls_loss + domain_loss" in da_phase:
            da_phase = da_phase.replace("total_loss = cls_loss + domain_loss", "total_loss = cls_loss + domain_loss + aux_domain_loss")
        elif "lambda_p * lmmd_loss" in da_phase:
            da_phase = da_phase.replace("total_loss = cls_loss + lambda_p * lmmd_loss", "total_loss = cls_loss + lambda_p * lmmd_loss + aux_domain_loss")
            
        code = da_split[0] + da_start + da_phase

    # 10. Accumulators in DA phase
    accum_logic = """            epoch_cls_loss    += cls_loss.item() * bs
            epoch_aux_domain_loss += aux_domain_loss.item() * bs
            epoch_ir_weight   += ir_weight_mean * bs"""
    code = code.replace("            epoch_cls_loss    += cls_loss.item() * bs", accum_logic)
    
    # 11. Normalize accumulators
    div_logic = """        epoch_cls_loss    /= max(total, 1)
        epoch_aux_domain_loss /= max(total, 1)
        epoch_ir_weight   /= max(total, 1)"""
    code = code.replace("        epoch_cls_loss    /= max(total, 1)", div_logic)

    # 12. Logging to losses dict
    if "lmmd" in code: # dsane / dsan
        code = re.sub(r'losses\s*=\s*\{.*?\}', 'losses = {"total": epoch_total_loss, "cls": epoch_cls_loss, "domain": epoch_aux_domain_loss, "lmmd": epoch_lmmd_loss}', code)
    else:
        code = re.sub(r'losses\s*=\s*\{.*?\}', 'losses = {"total": epoch_total_loss, "cls": epoch_cls_loss, "domain": epoch_domain_loss, "aux_dom": epoch_aux_domain_loss}', code)

    # 13. CSV Append
    if "dsan" in base:
        code = code.replace('"domain_loss": "0"', '"domain_loss": f"{epoch_aux_domain_loss:.6f}"')
    
    code = code.replace('"entropy_weight_mean": "NA"', '"entropy_weight_mean": f"{epoch_ir_weight:.4f}"')
    
    # 14. Fix the best acc hardcode
    code = code.replace('best acc: {0.0:.4f}', 'best acc: {ckpt.get(\"best_src_val_acc\", 0.0):.4f}')

    with open(out_path, "w") as f:
        f.write(code)

