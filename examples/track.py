from map_sort import Map_SORT, Detection, PixelMapper
import shapely.geometry as geo
import numpy as np
from tqdm import trange
import cv2
import imutils
import imutils.video
import matplotlib.pyplot as plt
 

video_path = "data/PETS09-S2L1.mp4"
map_path = "data/map.png"
dets_path = "data/det.txt"
point_map_path = "data/point_mapping.txt"
entry_polys_path = "data/entry_polys.txt"
output_path = "PETS09-S1L1-track.txt"
DISPLAY = True
SAVE = False

# Load the point mapping file
point_mapping = np.loadtxt(point_map_path, delimiter=",", dtype="int")
pixel_arr = point_mapping[:,:2]
map_arr = point_mapping[:,2:]

# Load the entry polys file
lines = []
with open(entry_polys_path, "r") as entry_file:
    lines = entry_file.readlines()
entry_polys = []
for line in lines:
    if line[0] == "#":
        continue
    values = [int(v) for v in line.split(",")]
    points = [(values[i], values[i+1]) for i in range(0,len(values), 2)]
    entry_polys.append(geo.Polygon(points))

# Setup tracking classes
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
map_img = cv2.imread(map_path)
vs = cv2.VideoCapture(video_path)
frame_total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
all_dets = np.loadtxt(dets_path, delimiter=',')
mapper = PixelMapper(pixel_arr, map_arr)
mot = Map_SORT(mapper, entry_polys, max_age=900, min_hits=3, iou_threshold=0.7, limit_entry=False)

output = []
for frame_num in trange(1, frame_total, unit="frame"):
    ret, frame = vs.read()
    if not ret or frame is None:
        break
    vis = frame.copy()
    map_vis = map_img.copy()

    # Load the detections for this frame in MOT Challenge format for this example
    dets = all_dets[all_dets[:,0] == frame_num]
    dets = [Detection(det[2:6], det[6]) for det in dets]

    # Run the tracking
    trackers = mot.step(dets)

    # Draw the output on the frame and map
    for trk, map_point, trk_id in trackers:
        box = trk.tlwh.astype("int")
        map_point = tuple([int(i) for i in np.squeeze(map_point)])
        output.append([frame_num, trk_id, box[0], box[1], box[2], box[3], map_point[0], map_point[1]])

        color = colors[trk_id%len(colors)]
        cv2.rectangle(vis, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
        cv2.circle(map_vis, map_point, 10, color, -1)
    if DISPLAY:
        vis = imutils.resize(vis, width=1080)
        cv2.imshow("Frame", vis)
        cv2.imshow("Map", map_vis)
        if cv2.waitKey(0)&0xFF == ord("q"):
            break

vs.release()
cv2.destroyAllWindows()

if SAVE:
    ret = np.array(output)
    np.savetxt(output_path, ret, delimiter=",", fmt="%d")