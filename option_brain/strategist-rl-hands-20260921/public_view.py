"""Filter before BOTH legacy and structured encoders; never pad a fake view."""
import copy

def restrict_raw(raw, public, known):
    out=copy.deepcopy(raw)
    enemies={e['id'] for e in public['enemies']}
    visible={(x,y) for x,y,_ in public['tiles']}
    floor={(i['x'],i['y']) for i in public['floor']}
    out['monsters']=[m for m in out['monsters'] if m['id'] in enemies]
    out['floor_items']=[i for i in out['floor_items'] if (i['x'],i['y']) in floor]
    # No appraisal or unidentified affix channel in this combat-only probe.
    for item in out['floor_items']:item['gear']=False
    exits={(e['x'],e['y']) for e in public['exits']}
    exits.update((e['x'],e['y']) for e in known.get('exits',{}).values())
    out['triggers']=[t for t in out.get('triggers',[]) if (t['x'],t['y']) in exits]
    out['progression_targets']=[t for t in out.get('progression_targets',[]) if (t['goal_x'],t['goal_y']) in visible]
    px,py=public['hero']['x'],public['hero']['y']
    out['missiles']=[m for m in out.get('missiles',[]) if bool(m.get('visible')) and
                     (px+int(m['tile_dx']),py+int(m['tile_dy'])) in visible]
    # No resource planner is used; retaining its full inventory appraisal would
    # create an unnecessary side channel even if the action were masked.
    out.pop('resource_state',None)
    return out

def restrict_map(raw_map, public, known, radius):
    out=copy.deepcopy(raw_map);side=radius*2+1
    px,py=public['hero']['x'],public['hero']['y']
    visible={(x,y) for x,y,_ in public['tiles']}
    enemies={(e['x'],e['y']) for e in public['enemies']}
    observed={tuple(map(int,k.split(','))):v for k,v in known['tiles'].items()}
    for key,values in list(out.items()):
        if not isinstance(values,(list,tuple)) or len(values)!=side*side:continue
        cleaned=[]
        for index,value in enumerate(values):
            p=(px+index%side-radius,py+index//side-radius)
            # Previously observed terrain may be remembered; dynamic occupants
            # and softwall state are never read through fog.
            if key=='monster':cleaned.append(int(p in enemies))
            elif p in visible:cleaned.append(value)
            elif key=='walkable':cleaned.append(int(observed.get(p,False)))
            else:cleaned.append(0)
        out[key]=cleaned
    return out
