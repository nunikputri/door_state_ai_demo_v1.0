# ==========================================================
# 🚗 Door-State AI Demo v1.3 — Real-Time Inference + Live Preview
# ==========================================================
import io, time, torch, os, datetime, numpy as np, json, cv2
from PIL import Image, ImageEnhance
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageDraw


# === Model Definition ===
class DoorStateCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# === Model Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load(r"door_state_cnn_final_script.pt", map_location=device)
model.eval()

transform = T.Compose([
    T.Resize((348, 348)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Define class names ===
class_names = [
    "s_00_all_closed", "s_01_front_left-open", "s_02_front_right-open",
    "s_03_front_left-open__front_right-open", "s_04_rear_left-open",
    "s_05_front_left-open__rear_left-open", "s_06_front_right-open__rear_left-open",
    "s_07_front_left-open__front_right-open__rear_left-open", "s_08_rear_right-open",
    "s_09_front_left-open__rear_right-open", "s_10_front_right-open__rear_right-open",
    "s_11_front_left-open__front_right-open__rear_right-open",
    "s_12_rear_left-open__rear_right-open", "s_13_front_left-open__rear_left-open__rear_right-open",
    "s_14_front_right-open__rear_left-open__rear_right-open",
    "s_15_front_left-open__front_right-open__rear_left-open__rear_right-open",
    "s_16_hood-open", "s_17_front_left-open__hood-open", "s_18_front_right-open__hood-open",
    "s_19_front_left-open__front_right-open__hood-open", "s_20_rear_left-open__hood-open",
    "s_21_front_left-open__rear_left-open__hood-open", "s_22_front_right-open__rear_left-open__hood-open",
    "s_23_front_left-open__front_right-open__rear_left-open__hood-open",
    "s_24_rear_right-open__hood-open", "s_25_front_left-open__rear_right-open__hood-open",
    "s_26_front_right-open__rear_right-open__hood-open",
    "s_27_front_left-open__front_right-open__rear_right-open__hood-open",
    "s_28_rear_left-open__rear_right-open__hood-open",
    "s_29_front_left-open__rear_left-open__rear_right-open__hood-open",
    "s_30_front_right-open__rear_left-open__rear_right-open__hood-open",
    "s_31_front_left-open__front_right-open__rear_left-open__rear_right-open__hood-open",
]

def parse_doors_from_label(label: str):
    """Return list of doors that are open based on prediction label."""
    doors = []
    for d in ["front_left", "front_right", "rear_left", "rear_right", "hood"]:
        if f"{d}-open" in label:
            doors.append(d)
    return doors


# === Selenium Setup ===
chrome_opts = Options()
# chrome_opts.add_argument("--start-maximized")
chrome_opts.add_argument("--window-size=1280,720")
chrome_opts.add_argument("--force-device-scale-factor=1")

driver = webdriver.Chrome(options=chrome_opts)
driver.get("https://euphonious-concha-ab5c5d.netlify.app/")

# Hide page clutter
# driver.execute_script("""
# (function(){
#   document.body.style.background='#000';
#   const c=document.querySelector('canvas');
#   if(c){
#     c.style.background='transparent';
#     c.style.position='absolute';
#     c.style.top='0';
#     c.style.left='0';
#     c.style.zIndex='10';
#   }
# })();
# """)

driver.execute_script("""
(function() {
  try {
    document.body.style.background = '#000';
    const canvas = document.querySelector('canvas');
    if (canvas) {
      canvas.style.background = 'transparent';
      canvas.style.position = 'absolute';
      canvas.style.top = '0';
      canvas.style.left = '0';
      canvas.style.zIndex = '10';
      canvas.style.opacity = '1.0';
    }
    const elements = document.querySelectorAll('div, header, nav, footer');
    elements.forEach(el => {
      try {
        const rect = el.getBoundingClientRect();
        if (rect.bottom < window.innerHeight * 0.2) {
          el.style.opacity = '0';
          el.style.pointerEvents = 'none';
        }
      } catch(e){}
    });
  } catch(e){}
})();
""")

# Inject status panel
sync_js = """
(function(){
  if (document.getElementById('car-status-panel')) return;
  const html = `
  <style>
    @keyframes pulseGlow {
      0%   { box-shadow: 0 0 0px rgba(16,185,129,0.0); }
      50%  { box-shadow: 0 0 20px rgba(16,185,129,0.8); }
      100% { box-shadow: 0 0 0px rgba(16,185,129,0.0); }
    }
  </style>
  <div id="car-status-panel" style="position:fixed;bottom:20px;left:20px;z-index:999999;
    width:250px;background:#f8fafc;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
    padding:10px 12px 24px 12px;font-family:Inter,system-ui;opacity:1;">
    <div id="predBox" style="background:#0b0b0b;color:white;border-radius:6px;
      padding:6px 8px;margin-bottom:8px;text-align:center;font-size:13px;position:relative;">
      Prediction: <span id="predLabel">-</span><br>
      <small id="predConf">Confidence: -</small>
      <div id="motionTag" style="position:absolute;bottom:6px;right:10px;font-size:12px;
      font-weight:500;color:#fff;background:#1e293b;padding:2px 6px;border-radius:6px;opacity:0.9;">
     </div>
    </div>
    <h3 style='margin:0 0 6px 0;font-size:15px;font-weight:600;'>Car Status</h3>
    <div id="statusContainer" style="display:flex;flex-direction:column;gap:5px;">
      ${["front_left","front_right","rear_left","rear_right","hood"].map(id=>`
        <div id="${id}-row" style="display:flex;justify-content:space-between;align-items:center;background:#fff;padding:4px 8px;border-radius:6px;">
          <span style='font-size:13px;text-transform:capitalize;'>${id.replace(/_/g,' ')}</span>
          <span id="${id}-status" style='font-size:11px;padding:3px 8px;border-radius:999px;background:#ef4444;color:white;'>Closed</span>
        </div>`).join('')}
    </div>
  </div>`;
  document.body.insertAdjacentHTML("beforeend", html);

  window.updateDoorStatusFromArray = function(doorArray){
    const doors = ["front_left","front_right","rear_left","rear_right","hood"];
    doors.forEach(d=>{
      const el = document.getElementById(d+'-status');
      if(!el) return;
      if(doorArray.includes(d)){
        el.textContent = "Open";
        el.style.background = "#10b981";
      } else {
        el.textContent = "Closed";
        el.style.background = "#ef4444";
      }
    });
  };
})();

"""
driver.execute_script(sync_js)



# === Inference Settings ===
CAR_CROP = (300, 0, 950, 550)
os.makedirs("captures", exist_ok=True)
last_frame = None
last_movement_time = time.time()
last_capture_time = 0
MOVEMENT_THRESHOLD = 8.0
STOP_DELAY = 3.0
CAPTURE_INTERVAL = 0.5
frame_idx = 0

# === Main Loop ===
try:
    while True:
        frame_idx += 1

        # --- Screenshot capture ---
        # try:
        #     png = driver.get_screenshot_as_png()
        #     img = Image.open(io.BytesIO(png)).convert("RGB").crop(CAR_CROP)
        #     img = ImageEnhance.Brightness(img).enhance(1.2)
        # except Exception as e:
        #     print(f"[WARN] Screenshot failed: {e}")
        #     time.sleep(0.5)
        #     continue
        # --- Screenshot capture + crop debug ---
        try:
            png = driver.get_screenshot_as_png()
            img_full = Image.open(io.BytesIO(png)).convert("RGB")
        
            # ✅ Draw the red rectangle where CAR_CROP is currently set
            draw = ImageDraw.Draw(img_full)
            draw.rectangle(CAR_CROP, outline="red", width=4)
        
            # # ✅ Show the full browser screenshot with the crop overlay
            # cv2.imshow("Full Browser View", cv2.cvtColor(np.array(img_full), cv2.COLOR_RGB2BGR))
        
            # # Optional: press 'c' to confirm crop view and continue
            # if cv2.waitKey(1) & 0xFF == ord('c'):
            #     print("[INFO] Crop confirmed. Proceeding to model view...")
        
            # Crop and adjust brightness for model inference
            img = img_full.crop(CAR_CROP)
            img = ImageEnhance.Brightness(img).enhance(1.1)
        
        except Exception as e:
            print(f"[WARN] Screenshot failed: {e}")
            time.sleep(0.5)
            continue


        # --- Motion detection ---
        np_img = np.array(img.resize((160, 90))).astype(np.float32)
        moving = False
        if last_frame is not None:
            diff = np.abs(np_img - last_frame).mean()
            if diff > MOVEMENT_THRESHOLD:
                moving = True
                last_movement_time = time.time()
        last_frame = np_img

        # --- Inference ---
        try:
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        out = model(tensor)
                else:
                    out = model(tensor)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                probs = torch.nn.functional.softmax(out, dim=1)
                conf_t, idx_t = probs.max(1)
                conf, idx = float(conf_t.item()), int(idx_t.item())
        except Exception as e:
            print(f"[WARN] Inference failed: {e}")
            time.sleep(0.5)
            continue

        # pred_label = class_names[idx] if conf >= 0.6 else "Uncertain"
        # doors_for_pred = [] if conf < 0.6 else parse_doors_from_label(pred_label)
        # doors_json = json.dumps(doors_for_pred)
        # Always predict the class (even for low confidence)
        pred_label = class_names[idx]
        
        # Always parse door states — even for uncertain predictions
        doors_for_pred = parse_doors_from_label(pred_label)
        doors_json = json.dumps(doors_for_pred)


        # --- Update browser panel ---
        driver.execute_script(f"""
        const lbl = document.getElementById('predLabel');
        const confBox = document.getElementById('predConf');
        const box = document.getElementById('predBox');
        if (lbl && confBox && box) {{
          lbl.textContent = "{pred_label}";
          confBox.textContent = "Confidence: {conf:.2f}";
          let bg;
          if ({conf:.2f} < 0.6) bg = '#0b0b0b';
          else if ({conf:.2f} < 0.9) bg = '#facc15';
          else bg = '#10b981';
          box.style.background = bg;
        }}
        if (window.updateDoorStatusFromArray)
          updateDoorStatusFromArray({doors_json});
        """)
        

        # --- Live Preview ---
        # frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # label_text = f"{pred_label} ({conf:.2f})"
        # color = (0, 255, 0) if conf >= 0.6 else (0, 165, 255)
        # cv2.putText(frame_bgr, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # cv2.imshow("🚘 Door-State Model View", frame_bgr)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("[INFO] Quit signal received.")
        #     break



        # # --- Save frames if moving ---
        # now = time.time()
        # if moving or (now - last_movement_time < STOP_DELAY):
        #     if now - last_capture_time >= CAPTURE_INTERVAL:
        #         last_capture_time = now
        #         ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #         fname = f"captures/{pred_label}_{ts}.png"
        #         img.save(fname)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[INFO] Exiting gracefully...")

finally:
    driver.quit()
    cv2.destroyAllWindows()
