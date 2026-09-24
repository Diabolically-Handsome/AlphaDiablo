"""Exact model158 actor, CPU inference only; no optimizer and no training."""
import sys
import torch
from torch import nn
from config import MODEL, MODEL_SHA, NETWORK_SOURCE, sha

def load_actor():
    if sha(MODEL)!=MODEL_SHA:raise RuntimeError('model158 identity changed')
    sys.path.insert(0,str(NETWORK_SOURCE/'train'))
    import leashed_ppo as L
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_net=nn.Sequential(nn.Linear(298,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
            self.context_adapter=L.NonlinearFusedContextAdapter(13012,64,
                excluded_observation_features=L.ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,device='cpu')
            self.action_net=nn.Linear(64,15)
        def forward(self,obs):
            if obs.ndim!=2 or obs.shape[1]!=13012:raise ValueError('original full input required')
            root=self.policy_net(obs[:,:298])
            return self.action_net(root+self.context_adapter(obs,root))
    doc=torch.load(MODEL,map_location='cpu',weights_only=True)
    if doc.get('iteration')!=158 or doc.get('observation_dimensions')!=13012 or doc.get('actions')!=15:
        raise RuntimeError('unexpected checkpoint contract')
    actor=Actor();state={k.removeprefix('actor.'):v for k,v in doc['state_dict'].items() if k.startswith('actor.')}
    actor.load_state_dict(state,strict=True)
    if not all(torch.equal(v,state[k]) for k,v in actor.state_dict().items()):raise RuntimeError('actor restore mismatch')
    actor.eval();actor.requires_grad_(False)
    return actor,dict(model_sha256=MODEL_SHA,iteration=doc['iteration'],parameters=sum(p.numel() for p in actor.parameters()),
                      tensors=len(state),device='cpu',optimizer_loaded=False,training=False)
