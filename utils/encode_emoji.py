import pickle
from collections import Counter

import emoji
from emoji.core import emojize

counter = Counter([])
print(counter)
for item in counter.items():
    print(item)

print(":".join([]))

emoji_obj = {"person": emoji.emojize(":couple:", use_aliases=True)}

coco_emoji_coco_name = ['person', 'bicycle', 'automobile', 'motorcycle', 'airplane',
                        'bus', 'train', 'delivery_truck', 'motor_boat', 'vertical_traffic_light',
                        'fire_engine', 'stop_sign', 'P_button', 'bench',
                         'bird', 'cat', 'dog', 'horse', 'ram', 'cow', 'elephant',
                         'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                         'necktie', 'briefcase', 'flying_disc', 'skis', 'snowboarder',
                         'baseball', 'kite', 'cricket_game', 'gloves',
                         'skateboard', 'person_surfing', 'tennis_racket', 'baby_bottle',
                         'wine_glass', 'cup_with_straw', 'fork_and_knife', 'kitchen_knife', 'spoon', 'cooked_rice',
                         'banana', 'red_apple', 'sandwich', 'tangerine', 'broccoli', 'carrot',
                         'hot_dog', 'pizza', 'doughnut', 'birthday_cake', 'chair', 'couch_and_lamp',
                         'potted_plant', 'bed', 'dining_table', 'toilet', 'television',
                         'laptop', 'mouse', 'video_game', 'keyboard', 'mobile_phone', 'microwave',
                         'oven', 'toaster', 'sink', 'refrigerator', 'books', 'alarm_clock',
                         'amphora', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

COCO_BBOX_LABEL_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                         'bus', 'train', 'truck', 'boat', 'traffic light',
                         'fire hydrant', 'stop sign', 'parking meter', 'bench',
                         'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                         'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                         'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                         'sports ball', 'kite', 'baseball bat', 'baseball glove',
                         'skateboard', 'surfboard', 'tennis racket', 'bottle',
                         'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                         'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                         'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                         'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

voc_emoji_name = ['airplane',  'bicycle', 'bird','motor_boat','baby_bottle', 'bus','car', 'cat' , 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


VOC_LABEL_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


d = {}
for name, emoji_name in zip(COCO_BBOX_LABEL_NAMES, coco_emoji_coco_name):
    d[name] = f":{emoji_name}:"
pickle.dump(d, open("./result/pkl/coco_emoji_names.pkl", 'wb'))

d = {}
for name, emoji_name in zip(VOC_LABEL_NAMES, voc_emoji_name):
    d[name] = f":{emoji_name}:"
pickle.dump(d, open("./result/pkl/voc_emoji_names.pkl", 'wb'))

print(emoji.demojize("👔"))

s = "1 :oven: ; 2 :refrigerator: ; 5 :tangerine: ; 1 :potted_plant: ; 1 :chair: ; 1 :banana:"
s2 = '1 :oven: ; 2 :refrigerator: ; 5 :tangerine: ; 1 :potted_plant: ; 1 :chair: ; 1 :banana:'
print(emoji.emojize('3 :person: ; 3 :cooked_rice: ; 1 :potted_plant: ; 1 :oven: ; 1 :spoon:'))
print(emoji.emojize(":vertical_traffic_light:"))
print(emoji.demojize("🎮"))
