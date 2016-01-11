import os
import pickle
HOME = os.path.expanduser("~")
PATH_LABELS = os.path.join(HOME, 'Dropbox/data/labels2.pkl')
metadata = pickle.load(open(PATH_LABELS, 'rb'))

invalid = [k for k,v in metadata['mprage'].items() if v is None]
valid = [k for k,v in metadata['mprage'].items() if v is not None]

print('apparently invalid but existing dataset')
for s in invalid:
    if os.path.exists(os.path.join('/local/matthiasg/mpr/', '%g' %s)):
        print(s)
# 3167, 3289, 3338, 2326, 4447, 4453

print('apparently valid but no existing dataset')
for s in valid:
    if not os.path.exists(os.path.join('/local/matthiasg/mpr/', '%g' %s)):
        print(s)
# 4141, 3388, 4961, 2003
