"""Synthetic first-update files/policy only. No real checkpoint, environment, or optimizer update."""
from copy import deepcopy
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

import analyze_first_update as analysis


class SyntheticPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.arange(15, dtype=torch.float32) / 20)
        self.value_bias = torch.nn.Parameter(torch.tensor(0.3))
        self.optimizer = SimpleNamespace(state_dict=lambda: {"state": {}, "param_groups": [{"params": [0, 1], "lr": 0.0001}]})
        self.calls = []

    def set_training_mode(self, enabled): self.train(enabled)

    def evaluate_actions(self, obs, actions, action_masks):
        self.calls.append(tuple(obs.shape))
        logits = self.weights[None, :] + obs[:, :1] * torch.arange(15, dtype=torch.float32)[None, :] / 100
        logits = logits.masked_fill(~torch.as_tensor(action_masks), -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        return (self.value_bias + obs[:, 1] * 0.05).reshape(-1, 1), dist.log_prob(actions), dist.entropy()


def policy_state(policy):
    return {key: value.detach().clone() for key, value in policy.state_dict().items()}


def gae(arrays, gamma=0.99, lam=0.95):
    result = np.zeros_like(arrays["values"])
    previous = 0
    for t in reversed(range(512)):
        nonterminal = 1.0 - (arrays["dones"].astype(np.float32) if t == 511 else arrays["episode_starts"][t+1])
        nv = arrays["last_values"] if t == 511 else arrays["values"][t+1]
        delta = arrays["rewards"][t] + gamma*nv*nonterminal - arrays["values"][t]
        previous = delta + gamma*lam*nonterminal*previous
        result[t] = previous
    arrays["advantages"] = result
    arrays["returns"] = result + arrays["values"]


def original_receipt(arrays, pending, actor_sha):
    r = np.array([row["transition_reward"] for row in pending], dtype=np.float64)
    c = np.array([row["combat_effect"] for row in pending], dtype=bool)
    adv = arrays["advantages"].reshape(-1).astype(np.float64)
    cr = r[c]
    result = {"rollout_end_timesteps": 2048, "collection_actor_sha256": actor_sha,
              "qualifies": False, "optimizer_steps": 8, "transition_reward_samples": 2048,
              "transition_reward_nonzero_samples": int(np.count_nonzero(r)),
              "transition_reward_positive_samples": int(np.count_nonzero(r>0)),
              "transition_reward_negative_samples": int(np.count_nonzero(r<0)),
              "transition_reward_sum": float(r.sum()), "transition_reward_abs_sum": float(np.abs(r).sum()),
              "transition_reward_mean": float(r.mean()), "transition_reward_variance": float(np.mean((r-r.mean())**2)),
              "combat_effect_samples": int(c.sum()), "combat_transition_reward_nonzero_samples": int(np.count_nonzero(cr)),
              "combat_transition_reward_positive_samples": int(np.count_nonzero(cr>0)),
              "combat_transition_reward_negative_samples": int(np.count_nonzero(cr<0)),
              "combat_transition_reward_sum": float(cr.sum()), "combat_transition_reward_abs_sum": float(np.abs(cr).sum()),
              "combat_positive_advantage_samples": int(np.count_nonzero(c & (adv>0))),
              "gae_advantage_samples": 2048, "gae_advantage_nonzero_samples": int(np.count_nonzero(adv)),
              "gae_advantage_mean": float(adv.mean()), "gae_advantage_variance": float(np.mean((adv-adv.mean())**2)),
              "requested_action_counts": np.bincount(arrays["actions"].reshape(-1).astype(np.int64), minlength=15).tolist(),
              "executed_action_counts": np.bincount([row["executed_action"] for row in pending], minlength=15).tolist(),
              "gae_recomputed_max_abs_delta": 0.0, "return_recomputed_max_abs_delta": 0.0}
    return result


@pytest.fixture(scope="module")
def capsule():
    policy = SyntheticPolicy()
    policy.set_training_mode(False)
    arrays = {key: np.zeros((512,4), dtype=np.float32) for key in analysis.ARRAY_NAMES}
    arrays["observations"] = np.zeros((512,4,13012), dtype=np.float32)
    arrays["observations"][:,:,0] = np.arange(512, dtype=np.float32)[:,None] / 512
    arrays["observations"][:,:,1] = np.arange(4, dtype=np.float32)[None,:]
    arrays["actions"] = ((np.arange(2048) % 14)+1).reshape(512,4,1).astype(np.int64)
    arrays["action_masks"] = np.ones((512,4,15), dtype=np.float32)
    arrays["action_masks"][:,:,0] = 0
    arrays["last_values"] = np.zeros(4, dtype=np.float32)
    arrays["dones"] = np.ones(4, dtype=bool)
    arrays["rewards"] = np.resize(np.array([1.0,0.5,-0.25,0.0], dtype=np.float32), (512,4))
    with torch.no_grad():
        for t in range(512):
            value, logs, _ = policy.evaluate_actions(torch.from_numpy(arrays["observations"][t]),
                torch.from_numpy(arrays["actions"][t,:,0]), arrays["action_masks"][t].astype(bool))
            arrays["values"][t] = value[:,0].numpy()
            arrays["log_probs"][t] = logs.numpy()
    gae(arrays)
    pending=[]
    for action, reward in zip(arrays["actions"].reshape(-1), arrays["rewards"].reshape(-1), strict=True):
        pending.append({"requested_action": int(action), "executed_action": int(action), "combat_effect": bool(action==9),
                        "transition_reward": float(reward), "worker_no_progress_timeout": False,
                        "no_progress_timeout_base_failure_reward": 0.0, "no_progress_timeout_additional_failure_reward": 0.0,
                        "no_progress_timeout_failure_reward": 0.0, "expected_buffer_reward": float(reward),
                        "time_limit_bootstrap": False, "time_limit_bootstrap_delta": 0.0})
    state=policy_state(policy)
    post=deepcopy(state); post["weights"][9]+=0.1; post["value_bias"]+=0.02
    descriptors={key:{"shape":list(value.shape),"dtype":str(value.dtype),"nbytes":value.nbytes,
                      "sha256":analysis.sha(value.tobytes())} for key,value in arrays.items()}
    contract={"artifact_scope":"candidate","implementation_sha256":"a"*64,"n_steps":512,"num_envs":4,"observation_shape":[13012],"action_n":15,
              "algorithm_recipe":{"clip_range":0.2,"gae_lambda":0.95,"normalize_advantage":True}}
    runtime={"torch_num_threads":1,"torch_num_interop_threads":torch.get_num_interop_threads(),"policy_training":False,
             "device":"cpu","gamma":0.99,"gae_lambda":0.95,"normalize_advantage":True}
    manifest={"schema":analysis.ARCHIVE_SCHEMA,"phase":"before","status":"DIAGNOSTIC_ONLY_NOT_PUBLISHABLE",
              "publication_eligible":False,"ordinary_resume_eligible":False,"ordinary_evaluation_eligible":False,
              "exact_trajectory_continuation":False,"implementation_sha256":"a"*64,"training_contract":contract,
              "resource_warm_start_receipt":{"schema":"diablogym-resource-warm-start/1","policy_sha256":analysis.tensor_digest(state),"synthetic":True},"counters":{key:0 for key in analysis.COUNTERS},
              "rollout_shape":[512,4],"array_order":"time-major","flatten_order":"t*n_envs+env","runtime":runtime,
              "actor_parameter_sha256":analysis.tensor_digest(state),"policy_tensor_sha256":analysis.tensor_digest(state),
              "before_archive_sha256":None,"arrays":descriptors,"members":{}}
    manifest["counters"].update(num_timesteps=2048,_total_timesteps=2048,_last_completed_ppo_rollout_steps=None)
    return {"arrays":arrays,"pending":pending,"state":state,"post":post,"manifest":manifest,
            "receipt":original_receipt(arrays,pending,manifest["actor_parameter_sha256"])}


def pt_bytes(value):
    stream=io.BytesIO(); torch.save(value,stream); return stream.getvalue()


def write_archive(path, manifest, members):
    manifest=deepcopy(manifest)
    manifest["members"]={name:{"sha256":analysis.sha(value),"bytes":len(value)} for name,value in members.items()}
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json",analysis.json_bytes(manifest))
        for name,value in members.items():output.writestr(name,value)
    return analysis.sha(path.read_bytes())


def files(tmp_path, capsule, *, arrays=None, pending=None, state=None, post=None, manifest_mutation=None, receipt_mutation=None):
    arrays=capsule["arrays"] if arrays is None else arrays
    pending=capsule["pending"] if pending is None else pending
    state=capsule["state"] if state is None else state
    post=capsule["post"] if post is None else post
    before=deepcopy(capsule["manifest"])
    before["arrays"]={key:{"shape":list(value.shape),"dtype":str(value.dtype),"nbytes":value.nbytes,"sha256":analysis.sha(value.tobytes())} for key,value in arrays.items()}
    before["policy_tensor_sha256"]=analysis.tensor_digest(state)
    before["actor_parameter_sha256"]=analysis.tensor_digest(state)
    if manifest_mutation:manifest_mutation(before)
    memory=io.BytesIO();np.savez_compressed(memory,**arrays)
    bp=tmp_path/"first_update_before.zip"
    before_sha=write_archive(bp,before,{"gae_snapshot.npz":memory.getvalue(),"pending_receipts.json":analysis.json_bytes(pending),
                                      "state/policy.pt":pt_bytes(state),"state/optimizer.pt":pt_bytes({"state":{},"param_groups":[{"params":[0,1],"lr":0.0001}]})})
    after=deepcopy(before);after.update(phase="after",before_archive_sha256=before_sha,
        policy_tensor_sha256=analysis.tensor_digest(post),actor_parameter_sha256=analysis.tensor_digest(post))
    after["runtime"]["policy_training"]=True
    after["counters"].update(_last_completed_ppo_rollout_steps=2048,_ppo_optimizer_steps_completed=8,
        _actor_optimizer_steps_completed=8,_worker_onpolicy_pg_joint_rollouts=1,_n_updates=1)
    receipt=original_receipt(arrays,pending,before["actor_parameter_sha256"])
    if receipt_mutation:receipt_mutation(receipt)
    ap=tmp_path/"first_update_after.zip"
    write_archive(ap,after,{"committed_receipts.json":analysis.json_bytes([receipt]),"state/policy.pt":pt_bytes(post),
                           "state/optimizer.pt":pt_bytes({"state":{0:{"step":torch.tensor(8.),"exp_avg":torch.ones(15),"exp_avg_sq":torch.ones(15)}},
                            "param_groups":[{"params":[0,1],"lr":0.0001}]})})
    init=tmp_path/"model_warm_start.zip";init.write_bytes(b"EXPLICIT_SYNTHETIC_LOADER_PAYLOAD_NOT_A_REAL_CHECKPOINT")
    return bp,ap,init


@pytest.fixture
def api(monkeypatch, capsule):
    record={"loads":0,"models":[],"loaded_parameter_flags":[]}
    def loader(payload):
        assert payload==b"EXPLICIT_SYNTHETIC_LOADER_PAYLOAD_NOT_A_REAL_CHECKPOINT"
        record["loads"]+=1
        # Deliberate loader RNG use must be invisible to the calling process.
        random.random();np.random.random();torch.rand(1)
        policy=SyntheticPolicy();policy.load_state_dict(capsule["state"])
        model=SimpleNamespace(policy=policy,device="cpu",get_env=lambda:None,observation_space=SimpleNamespace(shape=(13012,)),
            action_space=SimpleNamespace(n=15),diablogym_contract=deepcopy(capsule["manifest"]["training_contract"]),
            _resource_warm_start_receipt=deepcopy(capsule["manifest"]["resource_warm_start_receipt"]),gamma=0.99,gae_lambda=0.95,normalize_advantage=True,num_timesteps=0)
        record["loaded_parameter_flags"].append({name:parameter.requires_grad for name,parameter in policy.named_parameters()})
        record["models"].append(model);return model
    def receipt_validator(receipt,*,expected_samples):
        return expected_samples==2048 and receipt["optimizer_steps"]>0
    record["api"]=(loader,lambda policy,optimizer:analysis.tensor_digest(policy.state_dict()),receipt_validator,lambda:"a"*64)
    monkeypatch.setattr(analysis,"_production_api",lambda:record["api"])
    return record


def test_single_real_shape_synthetic_analysis_and_rng_restored(tmp_path,capsule,api):
    paths=files(tmp_path,capsule)
    py=random.getstate();n=np.random.get_state();t=torch.get_rng_state().clone();threads=torch.get_num_threads()
    result=analysis.analyze(*paths)
    assert result["status"]=="PASS_DIAGNOSTIC_ONLY" and result["actual_samples"]==2048
    assert result["committed_receipt"]["qualifies"] is False
    assert result["logprob_closure"]["max_abs_delta"]==0
    assert result["farm_dive_breakdown"]["available"] is False
    assert api["loads"]==1 and len(api["models"][0].policy.calls)==1024
    assert set(api["models"][0].policy.calls)=={(4,13012)}
    assert all(p.grad is None for p in api["models"][0].policy.parameters())
    assert {name:p.requires_grad for name,p in api["models"][0].policy.named_parameters()} == api["loaded_parameter_flags"][0]
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],n[1]) and torch.equal(torch.get_rng_state(),t)
    assert torch.get_num_threads()==threads
    assert result["objectives"]["delta"]["value_mse_fixed_returns"]!=0


def test_objectives_use_unbiased_global_std_and_global_combat_centering():
    a={"advantages":np.array([1.,2.,4.],np.float32),"log_probs":np.array([-1.,-2.,-3.],np.float32),"returns":np.array([2.,0.,1.],np.float32)}
    out={"log_probs":torch.tensor([-.9,-2.3,-2.9]),"values":torch.tensor([1.,0.,2.]),"entropy":torch.tensor([1.,2.,3.])}
    rows=[{"transition_reward":2.,"combat_effect":True},{"transition_reward":7.,"combat_effect":False},{"transition_reward":-1.,"combat_effect":False}]
    result=analysis.objectives(out,a,rows,0.2)
    import statistics
    advantages=[(x-statistics.mean([1,2,4]))/(statistics.stdev([1,2,4])+1e-8) for x in [1,2,4]]
    ratios=np.exp(np.array([.1,-.3,.1]))
    expected=-sum(min(x*r,x*min(1.2,max(.8,r))) for x,r in zip(advantages,ratios))/3
    assert result["gae_clipped_surrogate_loss"]==pytest.approx(expected,abs=1e-7)
    assert result["combat_centered_reference_loss"]==pytest.approx(-((2-2/3)*(-.9)+(0-2/3)*(-2.3)+(0-2/3)*(-2.9))/3,abs=1e-6)
    assert result["value_mse_fixed_returns"]==pytest.approx(2/3) and result["entropy_mean"]==2


@pytest.mark.parametrize("mutation,match",[
    (lambda m:m.update(array_order="env-major"),"ordering"),
    (lambda m:m.update(rollout_shape=[4,512]),"512 x 4"),
    (lambda m:m.update(publication_eligible=True),"publication"),
    (lambda m:m["counters"].update(_ppo_optimizer_steps_completed=1),"zero-update"),
    (lambda m:m["runtime"].update(torch_num_threads=True),"thread count"),
    (lambda m:m["runtime"].update(policy_training=True),"evaluation mode"),
])
def test_invalid_manifest_stops_before_model_load(tmp_path,capsule,api,mutation,match):
    paths=files(tmp_path,capsule,manifest_mutation=mutation)
    with pytest.raises(ValueError,match=match):analysis.analyze(*paths)
    assert api["loads"]==0


def test_original_receipt_rejection_not_overridden(tmp_path,capsule,api):
    paths=files(tmp_path,capsule)
    api["api"]=(api["api"][0],api["api"][1],lambda *a,**k:False,api["api"][3])
    with pytest.raises(ValueError,match="original formal receipt rejected"):analysis.analyze(*paths)
    assert api["loads"]==0


def test_receipt_aggregate_must_bind_actual_snapshot(tmp_path,capsule,api):
    paths=files(tmp_path,capsule,receipt_mutation=lambda r:r.update(transition_reward_sum=99.))
    with pytest.raises(ValueError,match="does not bind captured data"):analysis.analyze(*paths)
    assert api["loads"]==0


def test_time_major_pending_order_mismatch_rejected(tmp_path,capsule,api):
    rows=deepcopy(capsule["pending"]);rows[1],rows[4]=rows[4],rows[1]
    paths=files(tmp_path,capsule,pending=rows)
    with pytest.raises(ValueError,match="time-major action/receipt"):analysis.analyze(*paths)


def test_float32_integral_actions_keep_original_dtype(tmp_path,capsule,api):
    arrays=dict(capsule["arrays"]);arrays["actions"]=arrays["actions"].astype(np.float32)
    paths=files(tmp_path,capsule,arrays=arrays)
    result=analysis.analyze(*paths)
    assert result["actual_samples"]==2048
    assert analysis.read_archive(paths[0],"before")["manifest"]["arrays"]["actions"]["dtype"]=="float32"


def test_fractional_actions_rejected(tmp_path,capsule,api):
    arrays=dict(capsule["arrays"]);arrays["actions"]=arrays["actions"].astype(np.float32);arrays["actions"][0,0,0]=1.5
    paths=files(tmp_path,capsule,arrays=arrays)
    with pytest.raises(ValueError,match="exact integers"):analysis.analyze(*paths)


def test_actual_maskable_buffer_reset_add_dtype_without_model():
    from gymnasium import spaces
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    buffer=MaskableRolloutBuffer(2,spaces.Box(-1,1,shape=(13012,),dtype=np.float32),spaces.Discrete(15),device="cpu",n_envs=4)
    for t in range(2):
        buffer.add(np.zeros((4,13012),np.float32),np.array([1,2,3,4]),np.ones(4,np.float32),np.zeros(4,np.float32),
                   torch.zeros(4),torch.zeros(4),action_masks=np.ones((4,15),bool))
    assert buffer.full and buffer.actions.dtype==np.dtype(np.int64)
    np.testing.assert_array_equal(buffer.actions[:,:,0],np.tile([1,2,3,4],(2,1)))


def test_bootstrap_delta_forward_subtraction_is_not_inverted(tmp_path,capsule,api):
    arrays=dict(capsule["arrays"]);arrays["rewards"]=arrays["rewards"].copy()
    rows=deepcopy(capsule["pending"])
    raw=1e8;expected=np.float32(1.0);delta=np.float32(expected-raw)
    assert np.float32(raw)+delta!=expected
    rows[0].update(transition_reward=raw,expected_buffer_reward=float(expected),time_limit_bootstrap=True,time_limit_bootstrap_delta=float(delta))
    arrays["rewards"][0,0]=expected;gae(arrays)
    paths=files(tmp_path,capsule,arrays=arrays,pending=rows)
    before=analysis.read_archive(paths[0],"before");after=analysis.read_archive(paths[1],"after")
    analysis.validate_arrays(before,after)


def test_pre_logprob_tamper_fails_and_restores_rng(tmp_path,capsule,api):
    arrays=dict(capsule["arrays"]);arrays["log_probs"]=arrays["log_probs"].copy();arrays["log_probs"][0,0]+=.01
    paths=files(tmp_path,capsule,arrays=arrays)
    rng=torch.get_rng_state().clone();threads=torch.get_num_threads()
    with pytest.raises(ValueError,match="collection log-prob mismatch"):analysis.analyze(*paths)
    assert torch.equal(rng,torch.get_rng_state()) and torch.get_num_threads()==threads


def test_initializer_weight_mismatch_rejected(tmp_path,capsule,api):
    state=deepcopy(capsule["state"]);state["weights"][2]+=.5
    paths=files(tmp_path,capsule,state=state)
    with pytest.raises(ValueError,match="initialization differs"):analysis.analyze(*paths)
    assert api["loads"]==1


def test_after_extra_state_key_strict_load_rejected(tmp_path,capsule,api):
    post=deepcopy(capsule["post"]);post["unregistered_tensor"]=torch.zeros(1)
    paths=files(tmp_path,capsule,post=post)
    with pytest.raises(RuntimeError,match="Unexpected key"):analysis.analyze(*paths)


def test_archive_hash_mismatch_and_duplicate_members(tmp_path,capsule,api):
    paths=files(tmp_path,capsule)
    with zipfile.ZipFile(paths[0],"a") as archive:
        with pytest.warns(UserWarning,match="Duplicate name"):
            archive.writestr("pending_receipts.json",b"[]")
    with pytest.raises(ValueError,match="duplicates"):analysis.read_archive(paths[0],"before")


def test_archive_total_limit_before_decompression(tmp_path,capsule,monkeypatch):
    paths=files(tmp_path,capsule)
    monkeypatch.setattr(analysis,"MAX_ARCHIVE_BYTES",1)
    with pytest.raises(ValueError,match="expanded size"):analysis.read_archive(paths[0],"before")


def test_npz_total_limit_before_numpy_load(tmp_path,capsule,monkeypatch):
    paths=files(tmp_path,capsule)
    before=analysis.read_archive(paths[0],"before");after=analysis.read_archive(paths[1],"after")
    monkeypatch.setattr(analysis,"MAX_ARCHIVE_BYTES",1)
    with pytest.raises(ValueError,match="NPZ expanded size"):analysis.validate_arrays(before,after)


def test_no_output_overwrite(tmp_path):
    output=tmp_path/"analysis.json"
    analysis.write_output(output,{"status":"PASS_DIAGNOSTIC_ONLY"})
    old=output.read_bytes()
    with pytest.raises(ValueError,match="already exists"):analysis.write_output(output,{"wrong":True})
    assert output.read_bytes()==old


def test_cli_required_explicit_inputs():
    with pytest.raises(SystemExit):analysis.main([])


def test_actual_producer_consumer_synthetic_512x4_pair(tmp_path,capsule,api,monkeypatch):
    # Real callback and real archive reader/analyzer. Only policy/model and canonical
    # receipt validation are explicitly synthetic; no learn/optimizer/game calls.
    import rollout_diagnostics as capture
    import leashed_ppo
    from migrate_resource_candidate import ZERO_COUNTERS
    monkeypatch.setattr(leashed_ppo,"actor_parameter_sha256",api["api"][1])
    monkeypatch.setattr(leashed_ppo,"validate_worker_onpolicy_pg_receipt",api["api"][2])
    policy=SyntheticPolicy();policy.load_state_dict(capsule["state"]);policy.train(False)
    opt={"state":{},"param_groups":[{"params":[0,1],"lr":0.0001}]}
    policy.optimizer=SimpleNamespace(state_dict=lambda:deepcopy(opt))
    model=SimpleNamespace(**{key:0 for key in ZERO_COUNTERS})
    model.policy=policy;model.n_steps=512;model.n_envs=4;model._total_timesteps=2048
    model._last_completed_ppo_rollout_steps=None
    model._worker_onpolicy_pg_pending_receipts=[];model._worker_onpolicy_pg_rollout_receipts=[]
    model._worker_onpolicy_pg_audit_required=True;model._worker_onpolicy_pg_collection_actor_sha256=None
    model._calib_tripped=False;model._assert_critic_migration_contract=lambda:None
    model.diablogym_contract=deepcopy(capsule["manifest"]["training_contract"])
    model._resource_warm_start_receipt=deepcopy(capsule["manifest"]["resource_warm_start_receipt"])
    model.device="cpu";model.gamma=.99;model.gae_lambda=.95;model.normalize_advantage=True
    model.rollout_buffer=SimpleNamespace(full=False,generator_ready=False,_formal_gae_snapshot=None)
    cb=capture.FirstUpdateDiagnosticCallback(tmp_path,"a"*64);cb.init_callback(model)
    # Match the registered fixed synthetic evaluation runtime, restoring caller.
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        cb.on_training_start({"total_timesteps":2048},{})
        arrays=capsule["arrays"]
        model.rollout_buffer._formal_gae_snapshot={key:value.copy() for key,value in arrays.items()}
        for key,value in arrays.items():
            if key not in ("last_values","dones"):setattr(model.rollout_buffer,key,value.copy())
        model.rollout_buffer.full=True;model.num_timesteps=2048
        model._worker_onpolicy_pg_pending_receipts=deepcopy(capsule["pending"])
        cb.on_rollout_end()
        before=analysis.read_archive(tmp_path/"first_update_before.zip","before")
        policy.load_state_dict(capsule["post"]);policy.train(True)
        opt["state"]={0:{"step":torch.tensor(8.),"exp_avg":torch.ones(15),"exp_avg_sq":torch.ones(15)}}
        model._worker_onpolicy_pg_pending_receipts=[]
        model._worker_onpolicy_pg_rollout_receipts=[deepcopy(capsule["receipt"])]
        model._worker_onpolicy_pg_joint_rollouts=1;model._ppo_optimizer_steps_completed=8
        model._actor_optimizer_steps_completed=8;model._last_completed_ppo_rollout_steps=2048;model._n_updates=1
        model.rollout_buffer.generator_ready=True
        # The actual consumer must use the still-time-major sealed copy.
        model.rollout_buffer.observations=model.rollout_buffer.observations.swapaxes(0,1).reshape(2048,13012)
        cb.on_training_end()
        init=tmp_path/"model_warm_start.zip";init.write_bytes(b"EXPLICIT_SYNTHETIC_LOADER_PAYLOAD_NOT_A_REAL_CHECKPOINT")
        result=analysis.analyze(tmp_path/"first_update_before.zip",tmp_path/"first_update_after.zip",init)
        assert result["status"]=="PASS_DIAGNOSTIC_ONLY" and result["actual_samples"]==2048
        assert result["bindings"]["before_sha256"]==before["sha256"]
        assert result["logprob_closure"]["max_abs_delta"]==0.0
        assert json.loads((tmp_path/"rollout_diagnostic.json").read_text())["status"]=="COMPLETE"
        assert api["loads"]==1
    finally:torch.set_num_threads(threads)


def rewrite_zip(path, transform):
    with zipfile.ZipFile(path) as archive:
        members={name:archive.read(name) for name in archive.namelist()}
    transform(members)
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in members.items():archive.writestr(name,value)


def test_member_hash_is_checked_before_tensor_or_array_loading(tmp_path,capsule,api):
    paths=files(tmp_path,capsule)
    rewrite_zip(paths[0],lambda members:members.update({"pending_receipts.json":b"[]"}))
    with pytest.raises(ValueError,match="member hash/size"):analysis.analyze(*paths)
    assert api["loads"]==0


def test_after_must_bind_exact_before_archive(tmp_path,capsule,api):
    paths=files(tmp_path,capsule)
    def change(members):
        m=json.loads(members["manifest.json"]);m["before_archive_sha256"]="0"*64
        members["manifest.json"]=analysis.json_bytes(m)
    rewrite_zip(paths[1],change)
    with pytest.raises(ValueError,match="before/after archive binding"):analysis.analyze(*paths)
    assert api["loads"]==0


def test_gae_snapshot_tamper_rejected_even_with_rehashed_manifest(tmp_path,capsule,api):
    arrays=dict(capsule["arrays"]);arrays["advantages"]=arrays["advantages"].copy();arrays["advantages"][0,0]+=1
    paths=files(tmp_path,capsule,arrays=arrays)
    with pytest.raises(ValueError,match="GAE/return recurrence"):analysis.analyze(*paths)
    assert api["loads"]==0
