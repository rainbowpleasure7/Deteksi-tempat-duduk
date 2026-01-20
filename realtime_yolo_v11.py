import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Realtime YOLOv11 detection (image/video/camera)')
    p.add_argument('--model', required=True, help='Path to YOLO model (e.g. runs/detect/train/weights/best.pt)')
    p.add_argument('--source', required=True, help='Image file, folder, video file, or camera index (0,1)')
    p.add_argument('--conf', default=0.5, type=float, help='Confidence threshold')
    p.add_argument('--resize', default=None, help='Display/record resolution WxH (e.g. 1280x720)')
    p.add_argument('--record', action='store_true', help='Record output to demo_out.avi (requires --resize)')
    p.add_argument('--output', default=None, help='Optional output folder to save frames when headless')
    p.add_argument('--device', default=None, help='Device for inference (e.g. cpu or 0 or cuda:0)')
    p.add_argument('--half', action='store_true', help='Use FP16 half precision for inference if supported')
    p.add_argument('--scale', default=1.0, type=float, help='Scale factor for inference (0.5 = half resolution)')
    p.add_argument('--no-show', action='store_true', help='Do not display GUI windows (headless)')
    p.add_argument('--display-skip', default=1, type=int, help='Show GUI every N frames (reduce rendering overhead)')
    p.add_argument('--infer-skip', default=1, type=int, help='Run inference every N frames and reuse last results')
    p.add_argument('--persist-chairs', action='store_true', help='Persist detected chairs across frames when temporarily lost')
    p.add_argument('--persist-frames', default=10, type=int, help='Number of frames to keep a lost chair before removing')
    p.add_argument('--sticky-chairs', action='store_true', help='Make detected chairs stick permanently (never removed)')
    return p.parse_args()


def is_image_file(path):
    img_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG'}
    return os.path.splitext(path)[1] in img_ext


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter/union if union > 0 else 0.0


def main():
    args = parse_args()

    model_path = args.model
    src = args.source
    conf_thres = float(args.conf)
    out_folder = args.output

    if not os.path.exists(model_path):
        print('ERROR: model not found:', model_path)
        sys.exit(1)

    if out_folder and not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    model = YOLO(model_path, task='detect')
    labels = model.names

    # Determine source type
    source_type = None
    imgs_list = []

    if os.path.isdir(src):
        source_type = 'folder'
    elif os.path.isfile(src):
        if is_image_file(src):
            source_type = 'image'
        else:
            source_type = 'video'
    else:
        # try parse camera index
        try:
            cam_idx = int(src)
            source_type = 'camera'
            usb_idx = cam_idx
        except Exception:
            print('Invalid source:', src)
            sys.exit(1)

    # Prepare lists and capture
    if source_type == 'image':
        imgs_list = [src]
    elif source_type == 'folder':
        files = sorted(glob.glob(os.path.join(src, '*')))
        imgs_list = [f for f in files if is_image_file(f)]
    elif source_type in ('video', 'camera'):
        if source_type == 'video':
            cap_arg = src
        else:
            cap_arg = usb_idx
        cap = cv2.VideoCapture(cap_arg)
    else:
        print('Unsupported source type')
        sys.exit(1)

    resize = False
    if args.resize:
        try:
            rw, rh = args.resize.split('x')
            resW, resH = int(rw), int(rh)
            resize = True
        except Exception:
            print('Invalid --resize format. Use WxH like 1280x720')
            sys.exit(1)

    recorder = None
    if args.record:
        if not resize:
            print('Recording requires --resize to be set')
            sys.exit(1)
        record_name = 'demo_out.avi'
        fps = 25
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (resW, resH))

    img_count = 0
    frame_rate_buffer = []
    fps_avg_len = 120
    frame_idx = 0
    last_results = None
    chair_tracks = [] 
    next_chair_id = 1
    persist_enabled = bool(args.persist_chairs) if hasattr(args, 'persist_chairs') else False # digunakan untuk mengaktifkan fitur persist kursi sehingga tidak hilang saat tidak terdeteksi
    persist_frames = int(args.persist_frames) if hasattr(args, 'persist_frames') else 10
    sticky_enabled = bool(args.sticky_chairs) if hasattr(args, 'sticky_chairs') else False

    

    while True:
        t_start = time.perf_counter()

        if source_type in ('image', 'folder'):
            if img_count >= len(imgs_list):
                break
            frame_path = imgs_list[img_count]
            frame = cv2.imread(frame_path)
            if frame is None:
                print('Warning: failed to read', frame_path)
                img_count += 1
                continue
            img_count += 1
        else:
            ret, frame = cap.read()
            if not ret:
                break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        if frame is None:
            print('ERROR: empty frame, exiting')
            break

        # Keep a copy of original frame (before annotations) for separate display
        original_frame = frame.copy()

        # prepare scaled frame for faster inference (if requested)
        scale = float(args.scale) if hasattr(args, 'scale') else 1.0
        if scale <= 0 or scale > 1.0:
            scale = 1.0
        if scale != 1.0:
            inf_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            inf_frame = frame

        # Run inference every `infer_skip` frames (reuse last results otherwise)
        infer_skip = int(args.infer_skip) if hasattr(args, 'infer_skip') else 1
        if frame_idx % infer_skip == 0 or last_results is None:
            try:
                results = model.predict(source=inf_frame, conf=conf_thres, device=args.device if args.device else None, half=args.half, verbose=False)
            except TypeError:
                results = model.predict(source=inf_frame, conf=conf_thres, verbose=False)
            last_results = results
        else:
            results = last_results

        dets = results[0].boxes


        # Collect boxes per class so we can check overlaps (person vs chair)
        person_boxes = []
        detected_chairs = []  # detections this frame (to match/create tracks)

        for i in range(len(dets)):
            xyxy = dets[i].xyxy.cpu().numpy().squeeze()
            # map coordinates back to display frame if inference was done on scaled frame
            if scale != 1.0:
                try:
                    xyxy = (xyxy / scale)
                except Exception:
                    pass
            xyxy = xyxy.astype(int)
            xmin, ymin, xmax, ymax = xyxy.tolist()
            cls = int(dets[i].cls.item())
            conf = float(dets[i].conf.item())
            label = f"{labels[cls]}: {int(conf*100)}%"

            classname = labels[cls]
            cname = classname.lower()

            if cname == 'orang' or cname == 'person':
                color = (203, 192, 255)
                person_boxes.append([xmin, ymin, xmax, ymax])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
                font_scale = 0.3
                txt_sz, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                y0 = max(ymin, txt_sz[1] + 10)
                cv2.rectangle(frame, (xmin, y0-txt_sz[1]-10), (xmin+txt_sz[0], y0+base-10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, y0-7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
            elif cname in ('kursi', 'chair', 'seat'):
                detected_chairs.append({'bbox':[xmin, ymin, xmax, ymax], 'conf': conf})
            else:
                color = (0, 255, 0)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
                font_scale = 0.3
                txt_sz, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                y0 = max(ymin, txt_sz[1] + 10)
                cv2.rectangle(frame, (xmin, y0-txt_sz[1]-10), (xmin+txt_sz[0], y0+base-10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, y0-7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)

        # Chair tracking/matching
        track_iou_thr = 0.3
        # match detections to existing tracks
        used_det_idx = set()
        for d_idx, det in enumerate(detected_chairs):
            dbb = det['bbox']
            best_iou = 0.0
            best_t = None
            for t_idx, track in enumerate(chair_tracks):
                iou = calculate_iou(dbb, track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_t = t_idx
            if best_iou >= track_iou_thr and best_t is not None:
                # update existing track bbox and last seen
                chair_tracks[best_t]['bbox'] = dbb
                chair_tracks[best_t]['last_seen'] = frame_idx
                chair_tracks[best_t]['conf'] = det.get('conf', chair_tracks[best_t].get('conf', 0.0))
                used_det_idx.add(d_idx)

        # create new tracks for unmatched detections
        for d_idx, det in enumerate(detected_chairs):
            if d_idx in used_det_idx:
                continue
            chair_tracks.append({'id': next_chair_id, 'bbox': det['bbox'], 'last_seen': frame_idx, 'missed': 0, 'conf': det.get('conf', 0.0)})
            next_chair_id += 1

        # If persist is enabled but not sticky, increment missed and remove old tracks
        if persist_enabled and not sticky_enabled:
            remove_ids = []
            for track in chair_tracks:
                if track.get('last_seen', -1) != frame_idx:
                    track['missed'] = track.get('missed', 0) + 1
                if track.get('missed', 0) > persist_frames:
                    remove_ids.append(track['id'])
            chair_tracks = [t for t in chair_tracks if t['id'] not in remove_ids]

        # Draw all current tracks (sticky or persistent)
        for track in chair_tracks:
            xmin, ymin, xmax, ymax = track['bbox']
            # if persist but currently missed, draw faded; sticky stays normal
            if persist_enabled and not sticky_enabled and track.get('missed', 0) > 0:
                color = (80, 80, 200)
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            conf_val = track.get('conf', 0.0)
            label = f"kursi {track['id']} {int(conf_val*100)}%"
            font_scale = 0.3
            txt_sz, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            y0 = max(ymin, txt_sz[1] + 10)
            cv2.rectangle(frame, (xmin, y0-txt_sz[1]-10), (xmin+txt_sz[0], y0+base-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, y0-7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)

        # compute occupancy between persons and tracked chairs
        occupied_ids = set()
        iou_threshold = 0.1
        for pbox in person_boxes:
            for track in chair_tracks:
                if calculate_iou(pbox, track['bbox']) > iou_threshold:
                    occupied_ids.add(track['id'])
        total_chairs = len(chair_tracks)
        occupied_count = len(occupied_ids)
        empty_count = total_chairs - occupied_count

        # Display detected chair counts (total and empty)
        cv2.putText(frame, f'Kursi terdeteksi: {total_chairs}  Kosong: {empty_count}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # FPS calc
        t_stop = time.perf_counter()
        fps = 1.0 / (t_stop - t_start) if (t_stop - t_start) > 0 else 0.0
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_fps = float(np.mean(frame_rate_buffer))
        # Smaller FPS text for less visual clutter
        cv2.putText(frame, f'FPS: {avg_fps:0.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # Show or save (honor --no-show and display-skip to reduce overhead)
        no_show = bool(args.no_show) if hasattr(args, 'no_show') else False
        display_skip = int(args.display_skip) if hasattr(args, 'display_skip') else 1
        try:
            if recorder is not None:
                recorder.write(frame)

            if not no_show:
                if frame_idx % display_skip == 0:
                    if source_type == 'video':
                        cv2.namedWindow('Source - Original', cv2.WINDOW_NORMAL)
                        cv2.imshow('Source - Original', original_frame)
                        cv2.namedWindow('Detections - Annotated', cv2.WINDOW_NORMAL)
                        cv2.imshow('Detections - Annotated', frame)
                    else:
                        cv2.imshow('YOLOv11 Realtime', frame)

                if source_type in ('image', 'folder'):
                    key = cv2.waitKey()
                else:
                    key = cv2.waitKey(1)
            else:
                key = -1
        except cv2.error:
            # headless: save frame instead
            if out_folder:
                out_path = os.path.join(out_folder, f'detection_{img_count:06d}.jpg')
            else:
                out_path = f'detection_{img_count:06d}.jpg'
            cv2.imwrite(out_path, frame)
            print('Headless - saved', out_path)
            key = -1

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('p') or key == ord('P'):
            cv2.imwrite('capture.png', frame)

        # advance frame index (used for skipping inference/display)
        frame_idx += 1

    # cleanup
    if source_type in ('video', 'camera'):
        cap.release()
    if recorder is not None:
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
