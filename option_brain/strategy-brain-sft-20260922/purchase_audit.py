"""Native endless consumable stock receives a fresh identity on purchase."""
from collections import Counter

def owned(s):
    return [x for g in ('inventory','belt') for x in s[g] if not x['empty'] and x.get('name')!='Gold']

def purchased_item(before,after,quote):
    old=Counter(tuple(x['identity']) for x in owned(before));new=Counter(tuple(x['identity']) for x in owned(after))
    added=new-old;removed=old-new
    if removed or sum(added.values())!=1:return None
    token=next(iter(added));item=next(x for x in owned(after) if tuple(x['identity'])==token)
    keys=('name','quality','identified','location','min_damage','max_damage','armor','heal_kind','portal_scroll',
          'durability','max_durability','min_strength','min_dexterity','min_magic')
    keys+=tuple(k for k in quote if k.startswith('bonus_'))
    if any(item.get(k)!=quote.get(k) for k in keys):return None
    if after['hero']['gold']-before['hero']['gold']!=-quote['price']:return None
    return item
