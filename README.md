# Upbeat: your AI dance mirror

Upload any dance video, turn on your camera, and dance along. Upbeat tracks both bodies and tells you **live** what to fix: "Raise your left arm higher", "Bend your knees more", "You're ~0.3s behind".

## Run it
```bash
cd dance-coach && python3 -m http.server 8793
```
Open http://localhost:8793 in Chrome. The camera needs `localhost` or `https`.

## How it works
1. **Learn the choreo:** the reference video is scanned at 10 fps with MediaPipe Pose (full model). Each frame becomes a 33-point skeleton.
2. **Track you live:** your webcam runs through the lite pose model about 30 times a second, entirely in the browser.
3. **Compare:** each limb segment (upper arm, forearm, thigh, shin, torso) is compared by *direction*, so the match works for any body size or distance from the camera. You'll lag the video a little, so each live frame is checked against a short window of reference frames. The best-matching offset also drives the "behind / rushing" timing pill.
4. **Coach:**
   - Joint-angle rules turn the differences into plain-English cues (arm raise, elbow bend, leg lift, knee bend, lean).
   - Limbs turn green, yellow or red.
   - Optional voice coach reads the cues out loud.
5. **Review:** after a run you get:
   - an overall score
   - a score-over-time chart
   - a breakdown by body part
   - your 3 toughest moments, each with a one-click **Drill at 0.5×** loop
   - a download of your recorded take

## Tips
- **Mirror** flips the reference video left-to-right. Either way you're scored on copying what you see: the app detects whether the dancer faces the camera or has their back to it, frame by frame.
- Pause or scrub the reference to practice matching one frozen pose.
- **Hand gestures:** with the camera on, point left to rewind and point right to skip ahead (as you see yourself on screen). Keep your other fingers curled. Gestures are ignored while a scored run is going.
- Choose videos with one dancer and a full-body view.
