"""AI Face Attendance System - desktop GUI (Tkinter + OpenCV)."""
import csv
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import config
import database as db
from face_engine import FaceEngine, Gallery

GREEN, RED, BLUE, GREY = (0, 200, 0), (0, 0, 230), (230, 160, 0), (190, 190, 190)


class RegisterDialog(tk.Toplevel):
    """Asks for the person's details before face capture starts."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Register new person")
        self.resizable(False, False)
        self.result = None
        self.vars = {k: tk.StringVar() for k in ("name", "emp_id", "department")}
        frm = ttk.Frame(self, padding=16)
        frm.pack()
        fields = (("Full name *", "name"), ("ID / Roll no. *", "emp_id"), ("Department / Class", "department"))
        for i, (label, key) in enumerate(fields):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=4)
            e = ttk.Entry(frm, textvariable=self.vars[key], width=30)
            e.grid(row=i, column=1, pady=4, padx=(10, 0))
            if i == 0:
                e.focus_set()
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, pady=(12, 0), sticky="e")
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btns, text="Start capture", command=self._ok).pack(side="right")
        self.bind("<Return>", lambda e: self._ok())
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def _ok(self):
        data = {k: v.get().strip() for k, v in self.vars.items()}
        if not data["name"] or not data["emp_id"]:
            messagebox.showwarning("Missing info", "Name and ID are required.", parent=self)
            return
        if db.emp_id_exists(data["emp_id"]):
            messagebox.showwarning("Duplicate", "This ID is already registered.", parent=self)
            return
        self.result = data
        self.destroy()


class App:
    def __init__(self, root):
        self.root = root
        root.title("AI Face Attendance System")
        root.geometry("1180x700")
        root.minsize(980, 620)

        self.engine = None
        self.gallery = Gallery()
        self.mode = "idle"          # idle | attendance | register
        self.streak = {}
        self.marked = set()
        self.marked_date = ""
        self.reg = None
        self.photo = None
        self.cap = None

        self._build_ui()
        self._open_camera()
        threading.Thread(target=self._load_engine, daemon=True).start()
        root.protocol("WM_DELETE_WINDOW", self._quit)
        self._tick()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, width=640, height=480, bg="#111", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        side = ttk.Frame(main, padding=(12, 0, 0, 0), width=340)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        ttk.Label(side, text="Face Attendance", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        self.status = tk.StringVar(value="Loading AI models (first run downloads ~37 MB)...")
        ttk.Label(side, textvariable=self.status, wraplength=320, foreground="#444").pack(anchor="w", pady=(4, 10))

        self.btn_attend = ttk.Button(side, text="▶  Start attendance", command=self.toggle_attendance)
        self.btn_reg = ttk.Button(side, text="＋  Register new person", command=self.start_register)
        self.btn_reports = ttk.Button(side, text="📋  Attendance records / export", command=self.open_reports)
        self.btn_users = ttk.Button(side, text="👥  Manage people", command=self.open_users)
        self.action_buttons = [self.btn_attend, self.btn_reg]
        for b in (self.btn_attend, self.btn_reg, self.btn_reports, self.btn_users):
            b.pack(fill="x", pady=3, ipady=5)
        for b in self.action_buttons:
            b.state(["disabled"])

        ttk.Label(side, text="Today's attendance", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(14, 4))
        cols = ("time", "name", "status")
        self.today_tree = ttk.Treeview(side, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (80, 150, 70)):
            self.today_tree.heading(c, text=c.title())
            self.today_tree.column(c, width=w, anchor="w")
        self.today_tree.pack(fill="both", expand=True)
        self._refresh_today()

    def _open_camera(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
        if not self.cap.isOpened():
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def _load_engine(self):
        try:
            engine = FaceEngine()
            users, rows = db.load_gallery()
            self.root.after(0, lambda: self._engine_ready(engine, users, rows))
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: self._engine_failed(msg))

    def _engine_ready(self, engine, users, rows):
        self.engine = engine
        self.gallery.load(users, rows)
        for b in self.action_buttons:
            b.state(["!disabled"])
        self.status.set(f"Ready. {len(self.gallery)} registered. Register people, then start attendance.")

    def _engine_failed(self, msg):
        self.status.set("Failed to load AI models.")
        messagebox.showerror("Model error", msg)

    # ------------------------------------------------------------------ modes
    def toggle_attendance(self):
        if self.mode == "attendance":
            self._set_idle()
            return
        if len(self.gallery) == 0:
            messagebox.showinfo("No one registered", "Register at least one person first.")
            return
        today = datetime.now().strftime("%Y-%m-%d")
        self.marked, self.marked_date = db.marked_user_ids(today), today
        self.streak.clear()
        self.mode = "attendance"
        self.btn_attend.config(text="■  Stop attendance")
        self.btn_reg.state(["disabled"])
        self.status.set("Attendance running. Look at the camera.")

    def start_register(self):
        dlg = RegisterDialog(self.root)
        if not dlg.result:
            return
        self.reg = {**dlg.result, "embs": [], "last": 0.0, "thumb": None}
        self.mode = "register"
        self.btn_attend.state(["disabled"])
        self.btn_reg.state(["disabled"])
        self.status.set(f"Capturing face of {dlg.result['name']}. Look at the camera and slowly turn your head a little.")

    def _set_idle(self):
        self.mode = "idle"
        self.reg = None
        self.btn_attend.config(text="▶  Start attendance")
        for b in self.action_buttons:
            b.state(["!disabled"])
        self.status.set(f"Idle. {len(self.gallery)} registered.")

    # ------------------------------------------------------------------ video loop
    def _tick(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.flip(frame, 1)  # mirror view
                if self.engine:
                    try:
                        frame = self._process(frame)
                    except cv2.error:
                        pass
                self._show(frame)
        else:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                text="Camera not available.\nCheck CAMERA_INDEX in config.py", fill="white",
                font=("Segoe UI", 14), justify="center",
            )
        self.root.after(15, self._tick)

    def _show(self, frame):
        cw, ch = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        h, w = frame.shape[:2]
        s = min(cw / w, ch / h)
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo)

    def _process(self, frame):
        disp = frame.copy()                      # draw on a copy; embed from the clean frame
        faces = self.engine.detect(frame)
        if self.mode == "attendance":
            self._attendance(frame, disp, faces)
        elif self.mode == "register":
            self._register(frame, disp, faces)
        else:
            for f in faces:
                self._box(disp, f, GREY, "")
        return disp

    # ------------------------------------------------------------------ attendance
    def _attendance(self, frame, disp, faces):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.marked_date:            # midnight rollover
            self.marked, self.marked_date = db.marked_user_ids(today), today
            self._refresh_today()

        seen = set()
        for f in faces:
            if min(f[2], f[3]) < config.MIN_FACE_SIZE:
                self._box(disp, f, GREY, "Move closer")
                continue
            uid, score = self.gallery.identify(self.engine.embed(frame, f))
            if uid is None:
                self._box(disp, f, RED, "Unknown")
                continue
            seen.add(uid)
            name = self.gallery.users[uid]["name"]
            self.streak[uid] = self.streak.get(uid, 0) + 1
            if self.streak[uid] < config.CONSECUTIVE_FRAMES:
                self._box(disp, f, BLUE, f"{name} (verifying)")
                continue
            if uid not in self.marked:
                rec = db.mark_attendance(uid)
                self.marked.add(uid)
                if rec:
                    self.status.set(f"✔ {name} marked {rec['status']} at {rec['time']}")
                    self._refresh_today()
            self._box(disp, f, GREEN, f"{name} {score:.2f}" + ("  ✓" if uid in self.marked else ""))
        for uid in list(self.streak):
            if uid not in seen:
                del self.streak[uid]

    # ------------------------------------------------------------------ registration
    def _register(self, frame, disp, faces):
        r = self.reg
        if r is None:
            return
        n = len(r["embs"])
        total = config.REGISTER_SAMPLES
        if len(faces) == 0:
            self._banner(disp, "No face detected", RED)
        elif len(faces) > 1:
            self._banner(disp, "Only one person in frame, please", RED)
        else:
            f = faces[0]
            if min(f[2], f[3]) < config.MIN_FACE_SIZE or f[14] < 0.85:
                self._box(disp, f, RED, "")
                self._banner(disp, "Move closer / face the camera", RED)
            else:
                self._box(disp, f, BLUE, "")
                now = time.time()
                if now - r["last"] >= config.REGISTER_INTERVAL:
                    r["embs"].append(self.engine.embed(frame, f))
                    r["last"] = now
                    n += 1
                    if r["thumb"] is None:
                        x, y, w, h = [int(v) for v in f[:4]]
                        r["thumb"] = frame[max(y, 0): y + h, max(x, 0): x + w].copy()
        # progress bar
        H, W = disp.shape[:2]
        cv2.rectangle(disp, (20, H - 30), (W - 20, H - 14), (60, 60, 60), -1)
        cv2.rectangle(disp, (20, H - 30), (20 + int((W - 40) * n / total), H - 14), GREEN, -1)
        cv2.putText(disp, f"Capturing {n}/{total}", (22, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if n >= total:
            self._finish_register()

    def _finish_register(self):
        r, self.reg = self.reg, None
        try:
            uid = db.add_user(r["name"], r["emp_id"], r["department"], r["embs"])
            if r["thumb"] is not None and r["thumb"].size:
                cv2.imwrite(str(config.FACES_DIR / f"{uid}.jpg"), r["thumb"])
            users, rows = db.load_gallery()
            self.gallery.load(users, rows)
            msg = f"{r['name']} registered successfully."
        except Exception as e:
            msg = f"Registration failed: {e}"
        self._set_idle()
        self.root.after(50, lambda: messagebox.showinfo("Registration", msg))

    # ------------------------------------------------------------------ drawing helpers
    @staticmethod
    def _box(img, f, color, label):
        x, y, w, h = [int(v) for v in f[:4]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x, y - th - 10), (x + tw + 8, y), color, -1)
            cv2.putText(img, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    @staticmethod
    def _banner(img, text, color):
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), color, -1)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ------------------------------------------------------------------ side list
    def _refresh_today(self):
        self.today_tree.delete(*self.today_tree.get_children())
        for r in db.get_attendance(datetime.now().strftime("%Y-%m-%d")):
            self.today_tree.insert("", "end", values=(r["time"], r["name"], r["status"]))

    # ------------------------------------------------------------------ reports window
    def open_reports(self):
        win = tk.Toplevel(self.root)
        win.title("Attendance records")
        win.geometry("820x480")
        top = ttk.Frame(win, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Date (YYYY-MM-DD):").pack(side="left")
        var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(top, textvariable=var, width=12).pack(side="left", padx=6)

        cols = ("date", "time", "emp_id", "name", "department", "status")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c, w in zip(cols, (100, 80, 100, 220, 160, 80)):
            tree.heading(c, text=c.replace("_", " ").title())
            tree.column(c, width=w, anchor="w")
        tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        state = {"rows": [], "label": "all"}

        def load(day):
            if day:
                try:
                    datetime.strptime(day, "%Y-%m-%d")
                except ValueError:
                    messagebox.showwarning("Invalid date", "Use format YYYY-MM-DD.", parent=win)
                    return
            state["rows"], state["label"] = db.get_attendance(day), day or "all"
            tree.delete(*tree.get_children())
            for r in state["rows"]:
                tree.insert("", "end", values=[r[c] for c in cols])

        def export():
            if not state["rows"]:
                messagebox.showinfo("Nothing to export", "No records loaded.", parent=win)
                return
            path = filedialog.asksaveasfilename(
                parent=win, initialdir=str(config.EXPORT_DIR), defaultextension=".csv",
                initialfile=f"attendance_{state['label']}.csv", filetypes=[("CSV", "*.csv")],
            )
            if path:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    w.writerows(state["rows"])
                messagebox.showinfo("Exported", f"Saved to {path}", parent=win)

        ttk.Button(top, text="Load", command=lambda: load(var.get().strip())).pack(side="left")
        ttk.Button(top, text="All dates", command=lambda: load(None)).pack(side="left", padx=6)
        ttk.Button(top, text="Export CSV", command=export).pack(side="right")
        load(var.get())

    # ------------------------------------------------------------------ people window
    def open_users(self):
        win = tk.Toplevel(self.root)
        win.title("Registered people")
        win.geometry("720x420")
        cols = ("emp_id", "name", "department", "samples", "created_at")
        tree = ttk.Treeview(win, columns=cols, show="headings", selectmode="browse")
        for c, w in zip(cols, (110, 220, 150, 70, 140)):
            tree.heading(c, text=c.replace("_", " ").title())
            tree.column(c, width=w, anchor="w")
        tree.pack(fill="both", expand=True, padx=8, pady=8)
        ids = {}

        def load():
            tree.delete(*tree.get_children())
            ids.clear()
            for u in db.get_users():
                iid = tree.insert("", "end", values=[u[c] for c in cols])
                ids[iid] = u

        def delete():
            sel = tree.selection()
            if not sel:
                return
            u = ids[sel[0]]
            if not messagebox.askyesno("Delete", f"Delete {u['name']} and all their attendance records?", parent=win):
                return
            db.delete_user(u["id"])
            (config.FACES_DIR / f"{u['id']}.jpg").unlink(missing_ok=True)
            users, rows = db.load_gallery()
            self.gallery.load(users, rows)
            self.marked.discard(u["id"])
            load()
            self._refresh_today()

        ttk.Button(win, text="Delete selected", command=delete).pack(pady=(0, 8))
        load()

    # ------------------------------------------------------------------ exit
    def _quit(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    db.init_db()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam" if os.name != "nt" else "vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
