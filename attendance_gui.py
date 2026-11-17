import base64
import json
import os
import shutil
import time
import tkinter as tk
from datetime import date, datetime
from tkinter import messagebox, ttk

import cv2
import attendance as core


COLORS = {
    "page": "#F1F0E9",
    "paper": "#FFFFFF",
    "ink": "#1B2822",
    "muted": "#738078",
    "line": "#DCE1D9",
    "forest": "#193D32",
    "forest_light": "#28594A",
    "mint": "#D7E8DA",
    "orange": "#D78758",
    "red": "#B74F3D",
    "camera": "#10251E",
}


class AttendanceApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AttendX | Face Attendance")
        self.root.geometry("1220x790")
        self.root.minsize(1040, 700)
        self.root.configure(bg=COLORS["page"])

        self.people = core.load_people()
        self.detector = core.get_detector()
        self.recognizer = None
        self.camera = None
        self.camera_after_id = None
        self.camera_photo = None
        self.capture_state = None
        self.current_id = None
        self.current_confidence = None
        self.candidate_id = None
        self.candidate_frames = 0
        self.unknown_visible = False
        self.mode = tk.StringVar(value="test")
        self.threshold = tk.DoubleVar(value=65)
        self.replace_existing = tk.BooleanVar(value=False)

        self._setup_styles()
        self._build_shell()
        self._build_monitor_page()
        self._build_people_page()
        self._build_records_page()
        self._load_model()
        self.refresh_people()
        self.refresh_records()
        self.show_page("Monitor")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Attendance.Treeview",
            background=COLORS["paper"],
            foreground=COLORS["ink"],
            fieldbackground=COLORS["paper"],
            rowheight=34,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Attendance.Treeview.Heading",
            background=COLORS["page"],
            foreground=COLORS["muted"],
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map("Attendance.Treeview", background=[("selected", COLORS["mint"])])

    def _build_shell(self):
        shell = tk.Frame(self.root, bg=COLORS["page"])
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(shell, bg=COLORS["forest"], width=226)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="AttendX",
            bg=COLORS["forest"],
            fg="#FFFFFF",
            font=("Georgia", 20, "bold"),
        ).pack(anchor="w", padx=26, pady=(28, 2))
        tk.Label(
            sidebar,
            text="FACE ATTENDANCE  /  01",
            bg=COLORS["forest"],
            fg="#B4C8BB",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=28, pady=(0, 38))

        self.nav_buttons = {}
        for page, subtitle in (
            ("Monitor", "Camera and check-in"),
            ("People", "Roster and enrollment"),
            ("Records", "Excel attendance log"),
        ):
            button = tk.Button(
                sidebar,
                text=f"{page.upper()}\n{subtitle}",
                anchor="w",
                justify="left",
                padx=20,
                pady=13,
                bd=0,
                bg=COLORS["forest"],
                fg="#DDE8E0",
                activebackground=COLORS["forest_light"],
                activeforeground="#FFFFFF",
                font=("Segoe UI", 10),
                command=lambda selected=page: self.show_page(selected),
            )
            button.pack(fill="x", padx=12, pady=3)
            self.nav_buttons[page] = button

        tk.Frame(sidebar, bg="#496558", height=1).pack(fill="x", padx=24, pady=(28, 14))
        self.sidebar_status = tk.Label(
            sidebar,
            text="CAMERA OFFLINE",
            bg=COLORS["forest"],
            fg="#B4C8BB",
            font=("Segoe UI", 8, "bold"),
        )
        self.sidebar_status.pack(anchor="w", padx=28)
        tk.Label(
            sidebar,
            text="Local storage\nFace samples stay on this device",
            bg=COLORS["forest"],
            fg="#B4C8BB",
            justify="left",
            font=("Segoe UI", 9),
        ).pack(side="bottom", anchor="w", padx=28, pady=24)

        main = tk.Frame(shell, bg=COLORS["page"])
        main.pack(side="left", fill="both", expand=True)
        header = tk.Frame(main, bg=COLORS["page"], height=94)
        header.pack(fill="x", padx=34, pady=(20, 7))
        header.pack_propagate(False)
        heading = tk.Frame(header, bg=COLORS["page"])
        heading.pack(side="left", anchor="w")
        tk.Label(
            heading,
            text="LOCAL ATTENDANCE  /  FACE RECOGNITION",
            bg=COLORS["page"],
            fg=COLORS["forest_light"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        self.page_title = tk.Label(
            heading,
            text="Live monitor",
            bg=COLORS["page"],
            fg=COLORS["ink"],
            font=("Georgia", 26),
        )
        self.page_title.pack(anchor="w", pady=(4, 0))
        tk.Label(
            header,
            text=date.today().strftime("%A, %d %B %Y"),
            bg=COLORS["page"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(side="right", anchor="center", pady=27)

        self.page_host = tk.Frame(main, bg=COLORS["page"])
        self.page_host.pack(fill="both", expand=True, padx=34, pady=(4, 28))
        self.pages = {}

    def _build_monitor_page(self):
        page = tk.Frame(self.page_host, bg=COLORS["page"])
        self.pages["Monitor"] = page

        stats = tk.Frame(page, bg=COLORS["paper"], highlightbackground=COLORS["line"], highlightthickness=1)
        stats.pack(fill="x", pady=(0, 18))
        self.stat_people = self._stat(stats, "REGISTERED PEOPLE", "0")
        self.stat_today = self._stat(stats, "PRESENT TODAY", "0")
        self.stat_model = self._stat(stats, "RECOGNIZER", "Not trained")

        columns = tk.Frame(page, bg=COLORS["page"])
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=7)
        columns.grid_columnconfigure(1, weight=4)
        columns.grid_rowconfigure(0, weight=1)

        preview_panel = tk.Frame(columns, bg=COLORS["camera"])
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        preview_header = tk.Frame(preview_panel, bg=COLORS["camera"])
        preview_header.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(
            preview_header,
            text="CAMERA FEED",
            bg=COLORS["camera"],
            fg="#DCE9DF",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")
        self.camera_pill = tk.Label(
            preview_header,
            text="OFFLINE",
            bg="#30473D",
            fg="#DCE9DF",
            padx=9,
            pady=4,
            font=("Segoe UI", 8, "bold"),
        )
        self.camera_pill.pack(side="right")
        self.monitor_preview = tk.Label(
            preview_panel,
            text="Camera is off\nStart the camera to see live recognition",
            bg="#142A22",
            fg="#A9BEB1",
            justify="center",
            font=("Segoe UI", 11),
        )
        self.monitor_preview.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.preview_labels = [self.monitor_preview]

        control = tk.Frame(columns, bg=COLORS["paper"], highlightbackground=COLORS["line"], highlightthickness=1)
        control.grid(row=0, column=1, sticky="nsew")
        tk.Label(
            control,
            text="SESSION MODE",
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=20, pady=(18, 8))
        modes = tk.Frame(control, bg=COLORS["page"])
        modes.pack(fill="x", padx=18)
        self.test_mode_button = tk.Button(
            modes, text="Test", command=lambda: self.set_mode("test"), bd=0, padx=14, pady=9
        )
        self.test_mode_button.pack(side="left", fill="x", expand=True)
        self.live_mode_button = tk.Button(
            modes, text="Attendance", command=lambda: self.set_mode("attendance"), bd=0, padx=14, pady=9
        )
        self.live_mode_button.pack(side="left", fill="x", expand=True)

        tk.Label(
            control,
            text="RECOGNITION",
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=20, pady=(23, 8))
        self.recognition_name = tk.Label(
            control,
            text="Waiting for a face",
            bg=COLORS["paper"],
            fg=COLORS["ink"],
            anchor="w",
            font=("Georgia", 18),
        )
        self.recognition_name.pack(fill="x", padx=20)
        self.recognition_detail = tk.Label(
            control,
            text="Start the camera to begin",
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            anchor="w",
            font=("Segoe UI", 9),
        )
        self.recognition_detail.pack(fill="x", padx=20, pady=(5, 14))
        self.recognition_status = tk.Label(
            control,
            text="READY",
            bg=COLORS["mint"],
            fg=COLORS["forest"],
            anchor="w",
            padx=11,
            pady=8,
            font=("Segoe UI", 9, "bold"),
        )
        self.recognition_status.pack(fill="x", padx=20)

        self.camera_button = self._button(control, "Start camera", self.toggle_camera, primary=True)
        self.camera_button.pack(fill="x", padx=20, pady=(20, 8))
        self.mark_button = self._button(control, "Mark present", self.mark_present)
        self.mark_button.pack(fill="x", padx=20, pady=4)
        self.unknown_button = self._button(control, "Add unknown person", self.open_unknown_enrollment)
        self.unknown_button.pack(fill="x", padx=20, pady=4)
        self.train_button = self._button(control, "Train recognizer", self.train_recognizer)
        self.train_button.pack(fill="x", padx=20, pady=(4, 18))

    def _build_people_page(self):
        page = tk.Frame(self.page_host, bg=COLORS["page"])
        self.pages["People"] = page
        page.grid_columnconfigure(0, weight=4)
        page.grid_columnconfigure(1, weight=6)
        page.grid_rowconfigure(0, weight=1)

        form = tk.Frame(page, bg=COLORS["paper"], highlightbackground=COLORS["line"], highlightthickness=1)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        tk.Label(form, text="ADD A PERSON", bg=COLORS["paper"], fg=COLORS["forest_light"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=20, pady=(20, 5))
        tk.Label(form, text="Face enrollment", bg=COLORS["paper"], fg=COLORS["ink"], font=("Georgia", 19)).pack(anchor="w", padx=20, pady=(0, 16))
        self.person_id_var = tk.StringVar()
        self.person_name_var = tk.StringVar()
        self.sample_count_var = tk.StringVar(value="30")
        self._field(form, "Numeric ID", self.person_id_var)
        self._field(form, "Full name", self.person_name_var)
        self._field(form, "Samples", self.sample_count_var)
        tk.Checkbutton(
            form,
            text="Replace this ID's existing samples",
            variable=self.replace_existing,
            bg=COLORS["paper"],
            fg=COLORS["ink"],
            activebackground=COLORS["paper"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=18, pady=(2, 8))
        self.capture_button = self._button(form, "Start camera first", self.start_enrollment, primary=True)
        self.capture_button.pack(fill="x", padx=20, pady=(8, 8))
        self.capture_progress = tk.Label(form, text="No samples being captured", bg=COLORS["paper"], fg=COLORS["muted"], font=("Segoe UI", 9))
        self.capture_progress.pack(anchor="w", padx=20, pady=(3, 18))

        roster = tk.Frame(page, bg=COLORS["paper"], highlightbackground=COLORS["line"], highlightthickness=1)
        roster.grid(row=0, column=1, sticky="nsew")
        tk.Label(roster, text="CURRENT ROSTER", bg=COLORS["paper"], fg=COLORS["forest_light"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=20, pady=(20, 4))
        tk.Label(roster, text="Registered people", bg=COLORS["paper"], fg=COLORS["ink"], font=("Georgia", 19)).pack(anchor="w", padx=20, pady=(0, 12))
        self.people_tree = ttk.Treeview(roster, columns=("id", "name", "samples"), show="headings", style="Attendance.Treeview")
        self.people_tree.heading("id", text="ID")
        self.people_tree.heading("name", text="NAME")
        self.people_tree.heading("samples", text="SAMPLES")
        self.people_tree.column("id", width=70, anchor="center")
        self.people_tree.column("name", width=220)
        self.people_tree.column("samples", width=100, anchor="center")
        self.people_tree.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        self.people_preview = tk.Label(roster, text="Live preview appears here during enrollment", bg=COLORS["camera"], fg="#A9BEB1", height=5, font=("Segoe UI", 9))
        self.people_preview.pack(fill="x", padx=20, pady=(0, 20))
        self.preview_labels.append(self.people_preview)

    def _build_records_page(self):
        page = tk.Frame(self.page_host, bg=COLORS["page"])
        self.pages["Records"] = page
        toolbar = tk.Frame(page, bg=COLORS["page"])
        toolbar.pack(fill="x", pady=(0, 14))
        tk.Label(toolbar, text="Saved to attendance_data/attendance.xlsx", bg=COLORS["page"], fg=COLORS["muted"], font=("Segoe UI", 10)).pack(side="left")
        self._button(toolbar, "Refresh", self.refresh_records).pack(side="right", padx=(8, 0))
        self._button(toolbar, "Open in Excel", self.open_excel, primary=True).pack(side="right")

        table_frame = tk.Frame(page, bg=COLORS["paper"], highlightbackground=COLORS["line"], highlightthickness=1)
        table_frame.pack(fill="both", expand=True)
        self.records_tree = ttk.Treeview(table_frame, columns=("date", "time", "id", "name"), show="headings", style="Attendance.Treeview")
        for column, title, width in (
            ("date", "DATE", 150),
            ("time", "TIME", 130),
            ("id", "ID", 100),
            ("name", "NAME", 300),
        ):
            self.records_tree.heading(column, text=title)
            self.records_tree.column(column, width=width, anchor="w" if column == "name" else "center")
        self.records_tree.pack(fill="both", expand=True, padx=16, pady=16)
        self.records_status = tk.Label(page, text="", bg=COLORS["page"], fg=COLORS["muted"], anchor="w", font=("Segoe UI", 9))
        self.records_status.pack(fill="x", pady=(9, 0))

    def _stat(self, parent, title, value):
        cell = tk.Frame(parent, bg=COLORS["paper"])
        cell.pack(side="left", fill="x", expand=True, padx=20, pady=15)
        tk.Label(cell, text=title, bg=COLORS["paper"], fg=COLORS["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        value_label = tk.Label(cell, text=value, bg=COLORS["paper"], fg=COLORS["ink"], font=("Georgia", 17))
        value_label.pack(anchor="w", pady=(4, 0))
        return value_label

    def _button(self, parent, text, command, primary=False):
        background = COLORS["forest"] if primary else COLORS["page"]
        foreground = "#FFFFFF" if primary else COLORS["ink"]
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=COLORS["forest_light"] if primary else COLORS["mint"],
            activeforeground="#FFFFFF" if primary else COLORS["ink"],
            disabledforeground="#A8B1AB",
            bd=0,
            padx=14,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )

    def _field(self, parent, title, variable):
        tk.Label(parent, text=title, bg=COLORS["paper"], fg=COLORS["muted"], font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(8, 4))
        entry = tk.Entry(parent, textvariable=variable, relief="flat", bg=COLORS["page"], fg=COLORS["ink"], insertbackground=COLORS["forest"], font=("Segoe UI", 11))
        entry.pack(fill="x", padx=20, ipady=8)

    def show_page(self, page):
        for current in self.pages.values():
            current.pack_forget()
        self.pages[page].pack(fill="both", expand=True)
        self.page_title.configure(text={"Monitor": "Live monitor", "People": "People and enrollment", "Records": "Attendance records"}[page])
        for name, button in self.nav_buttons.items():
            active = name == page
            button.configure(bg=COLORS["forest_light"] if active else COLORS["forest"], fg="#FFFFFF" if active else "#DDE8E0")
        if page == "Records":
            self.refresh_records()
        if page == "People":
            self.refresh_people()

    def set_mode(self, mode):
        self.mode.set(mode)
        for button, selected in ((self.test_mode_button, mode == "test"), (self.live_mode_button, mode == "attendance")):
            button.configure(
                bg=COLORS["forest"] if selected else COLORS["page"],
                fg="#FFFFFF" if selected else COLORS["ink"],
                activebackground=COLORS["forest_light"],
            )
        self._update_recognition()

    def _load_model(self):
        self.people = core.load_people()
        if not core.MODEL_FILE.exists():
            self.recognizer = None
            self.stat_model.configure(text="Not trained")
            return
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(str(core.MODEL_FILE))
            self.recognizer = recognizer
            self.stat_model.configure(text="Ready")
        except cv2.error as error:
            self.recognizer = None
            self.stat_model.configure(text="Model error")
            self._set_status(f"Could not load recognizer: {error}", error=True)

    def toggle_camera(self):
        if self.camera is not None:
            self.stop_camera()
            return
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror("Camera unavailable", "Could not open the webcam. Check Windows camera permission and close other camera apps.")
            return
        self.camera = camera
        self.camera_button.configure(text="Stop camera", bg=COLORS["red"], activebackground="#933D32")
        self.camera_pill.configure(text="LIVE", bg=COLORS["forest_light"], fg="#FFFFFF")
        self.sidebar_status.configure(text="CAMERA LIVE", fg="#D7E8DA")
        self._set_status("Camera running", error=False)
        self._update_camera()

    def stop_camera(self):
        if self.camera_after_id is not None:
            self.root.after_cancel(self.camera_after_id)
            self.camera_after_id = None
        if self.capture_state is not None:
            self.cancel_enrollment(show_message=False)
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.camera_button.configure(text="Start camera", bg=COLORS["forest"], activebackground=COLORS["forest_light"])
        self.camera_pill.configure(text="OFFLINE", bg="#30473D", fg="#DCE9DF")
        self.sidebar_status.configure(text="CAMERA OFFLINE", fg="#B4C8BB")
        self.current_id = None
        self.current_confidence = None
        self.candidate_id = None
        self.candidate_frames = 0
        self.unknown_visible = False
        self._update_recognition()
        for label in self.preview_labels:
            label.configure(image="", text="Camera is off")
        self.camera_photo = None

    def _update_camera(self):
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self._set_status("Unable to read from camera", error=True)
            self.stop_camera()
            return

        gray, faces = core.find_faces(frame, self.detector)
        self.current_id = None
        self.current_confidence = None
        self.unknown_visible = False
        if len(faces) == 1:
            x, y, width, height = faces[0]
            face = cv2.equalizeHist(gray[y : y + height, x : x + width])
            face = cv2.resize(face, core.FACE_SIZE)
            if self.recognizer is not None:
                predicted_id, confidence = self.recognizer.predict(face)
                person_id = str(predicted_id)
                self.current_confidence = confidence
                if confidence <= self.threshold.get() and person_id in self.people:
                    self.current_id = person_id
                else:
                    self.unknown_visible = True
            else:
                self.unknown_visible = True
            color = (72, 196, 128) if self.current_id else (71, 105, 227)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

        if self.current_id:
            if self.current_id == self.candidate_id:
                self.candidate_frames += 1
            else:
                self.candidate_id = self.current_id
                self.candidate_frames = 1
        else:
            self.candidate_id = None
            self.candidate_frames = 0

        self._collect_face_sample(gray, faces)
        self._update_recognition()
        self._show_frame(frame)
        self.camera_after_id = self.root.after(33, self._update_camera)

    def _show_frame(self, frame):
        preview = cv2.resize(frame, (720, 405), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", preview)
        if not ok:
            return
        image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        self.camera_photo = tk.PhotoImage(master=self.root, data=image_data)
        for label in self.preview_labels:
            label.configure(image=self.camera_photo, text="")

    def _update_recognition(self):
        mode_is_test = self.mode.get() == "test"
        if self.current_id and self.candidate_frames >= 5:
            name = self.people.get(self.current_id, "Unknown")
            self.recognition_name.configure(text=name, fg=COLORS["forest"])
            score = "" if self.current_confidence is None else f"LBPH distance {self.current_confidence:.1f}"
            self.recognition_detail.configure(text=f"ID {self.current_id}  /  {score}")
            if mode_is_test:
                self.recognition_status.configure(text="TEST MATCH · NO RECORD WRITTEN", bg=COLORS["mint"], fg=COLORS["forest"])
            else:
                self.recognition_status.configure(text="CHECK NAME BEFORE CONFIRMING", bg="#F5E5D8", fg="#8A4C2B")
            self.mark_button.configure(state="disabled" if mode_is_test else "normal")
        elif self.unknown_visible:
            self.recognition_name.configure(text="Unknown face", fg=COLORS["red"])
            self.recognition_detail.configure(text="No confident match in the current roster")
            self.recognition_status.configure(text="CAN BE ENROLLED", bg="#F5E5D8", fg="#8A4C2B")
            self.mark_button.configure(state="disabled")
        else:
            self.recognition_name.configure(text="Waiting for a face", fg=COLORS["ink"])
            self.recognition_detail.configure(text="Hold one face in view for a moment")
            self.recognition_status.configure(text="READY" if self.camera else "CAMERA OFFLINE", bg=COLORS["mint"], fg=COLORS["forest"])
            self.mark_button.configure(state="disabled")
        self.unknown_button.configure(state="normal" if self.unknown_visible and not mode_is_test else "disabled")

    def mark_present(self):
        if self.mode.get() != "attendance" or not self.current_id or self.candidate_frames < 5:
            return
        try:
            attended = core.load_attendance_for_today()
            created = core.record_attendance(self.current_id, self.people[self.current_id], attended)
        except PermissionError:
            messagebox.showerror("Excel is open", "Close attendance.xlsx in Excel, then try marking present again.")
            return
        except (OSError, KeyError) as error:
            messagebox.showerror("Could not record attendance", str(error))
            return
        if created:
            self._set_status(f"Recorded {self.people[self.current_id]} in Excel", error=False)
            self.refresh_records()
        else:
            self._set_status(f"{self.people[self.current_id]} is already recorded today", error=False)

    def open_unknown_enrollment(self):
        self.show_page("People")
        self.person_id_var.set("")
        self.person_name_var.set("")
        self.capture_progress.configure(text="Enter a new ID and name, then capture samples")

    def start_enrollment(self):
        if self.camera is None:
            messagebox.showinfo("Start camera", "Start the camera on the Monitor page first, then return here to capture samples.")
            return
        if self.capture_state is not None:
            self.cancel_enrollment(show_message=True)
            return
        person_id = self.person_id_var.get().strip()
        name = self.person_name_var.get().strip()
        try:
            sample_count = int(self.sample_count_var.get())
        except ValueError:
            messagebox.showerror("Invalid sample count", "Enter a whole number of samples.")
            return
        if not person_id.isdigit() or int(person_id) < 1:
            messagebox.showerror("Invalid ID", "Use a positive numeric ID.")
            return
        person_id = str(int(person_id))
        if not name:
            messagebox.showerror("Name required", "Enter the person's name.")
            return
        if sample_count < 1 or sample_count > 200:
            messagebox.showerror("Invalid sample count", "Choose between 1 and 200 samples.")
            return
        if person_id in self.people and not self.replace_existing.get():
            messagebox.showerror("ID already exists", "Select the replace option to recapture this person's samples.")
            return
        if person_id in self.people and not messagebox.askyesno("Replace face samples?", f"Replace saved samples for {self.people[person_id]} (ID {person_id}) after capture completes?"):
            return

        capture_dir = core.FACES_DIR / f".{person_id}_gui_capture"
        if capture_dir.exists():
            shutil.rmtree(capture_dir)
        capture_dir.mkdir(parents=True, exist_ok=True)
        self.capture_state = {
            "id": person_id,
            "name": name,
            "count": sample_count,
            "saved": 0,
            "directory": capture_dir,
            "last_capture": 0.0,
        }
        self.capture_button.configure(text="Cancel capture", bg=COLORS["red"])
        self.capture_progress.configure(text=f"Captured 0 of {sample_count} samples · show one face")

    def _collect_face_sample(self, gray, faces):
        state = self.capture_state
        if state is None:
            return
        now = time.monotonic()
        if len(faces) != 1 or now - state["last_capture"] < 0.18:
            return
        x, y, width, height = faces[0]
        face = cv2.equalizeHist(gray[y : y + height, x : x + width])
        face = cv2.resize(face, core.FACE_SIZE)
        sample_path = state["directory"] / f"{state['saved']:03}.png"
        if not cv2.imwrite(str(sample_path), face):
            self.cancel_enrollment(show_message=False)
            messagebox.showerror("Capture failed", "Could not save a face sample.")
            return
        state["saved"] += 1
        state["last_capture"] = now
        self.capture_progress.configure(text=f"Captured {state['saved']} of {state['count']} samples · vary angle and expression slightly")
        if state["saved"] >= state["count"]:
            self._finish_enrollment()

    def _finish_enrollment(self):
        state = self.capture_state
        self.capture_state = None
        target = core.FACES_DIR / state["id"]
        if target.exists():
            shutil.rmtree(target)
        state["directory"].rename(target)
        self.people[state["id"]] = state["name"]
        core.DATA_DIR.mkdir(parents=True, exist_ok=True)
        core.PEOPLE_FILE.write_text(json.dumps(self.people, indent=2), encoding="utf-8")
        self.capture_button.configure(text="Capture face samples", bg=COLORS["forest"])
        self.capture_progress.configure(text="Samples saved. Training recognizer…")
        self.root.update_idletasks()
        try:
            core.train_model()
            self._load_model()
            self.refresh_people()
            self.capture_progress.configure(text=f"{state['name']} enrolled · recognizer updated")
            self._set_status(f"Enrolled {state['name']} and trained model", error=False)
            messagebox.showinfo("Enrollment complete", f"{state['name']} is enrolled. They can be recognized by this session now.")
        except (OSError, RuntimeError, cv2.error) as error:
            self.capture_progress.configure(text="Samples saved, but model training failed")
            messagebox.showerror("Training failed", str(error))

    def cancel_enrollment(self, show_message):
        state = self.capture_state
        if state is None:
            return
        self.capture_state = None
        shutil.rmtree(state["directory"], ignore_errors=True)
        self.capture_button.configure(text="Capture face samples", bg=COLORS["forest"])
        self.capture_progress.configure(text="Capture cancelled; previous samples were kept")
        if show_message:
            self._set_status("Enrollment cancelled", error=False)

    def train_recognizer(self):
        try:
            core.train_model()
            self._load_model()
            self._set_status("Recognizer trained", error=False)
            messagebox.showinfo("Training complete", "The recognizer has been updated from the current roster.")
        except (OSError, RuntimeError, cv2.error) as error:
            messagebox.showerror("Training failed", str(error))

    def refresh_people(self):
        self.people = core.load_people()
        if not hasattr(self, "people_tree"):
            return
        for row in self.people_tree.get_children():
            self.people_tree.delete(row)
        for person_id, name in sorted(self.people.items(), key=lambda item: int(item[0])):
            sample_dir = core.FACES_DIR / person_id
            sample_count = len(list(sample_dir.glob("*.png"))) if sample_dir.exists() else 0
            self.people_tree.insert("", "end", values=(person_id, name, sample_count))
        self.stat_people.configure(text=str(len(self.people)))

    def _read_records(self):
        if not core.ATTENDANCE_FILE.exists():
            return []
        workbook = core.load_workbook(core.ATTENDANCE_FILE, read_only=True, data_only=True)
        try:
            return list(workbook["Attendance"].iter_rows(min_row=2, values_only=True))
        finally:
            workbook.close()

    def refresh_records(self):
        if not hasattr(self, "records_tree"):
            return
        for row in self.records_tree.get_children():
            self.records_tree.delete(row)
        try:
            rows = self._read_records()
        except PermissionError:
            self.records_status.configure(text="Excel has the workbook locked. Close it to refresh records.", fg=COLORS["red"])
            return
        for row in reversed(rows):
            if not row or row[0] is None:
                continue
            recorded_date, recorded_time, person_id, name = (list(row) + [None] * 4)[:4]
            if isinstance(recorded_date, (datetime, date)):
                recorded_date = recorded_date.strftime("%Y-%m-%d")
            if hasattr(recorded_time, "strftime"):
                recorded_time = recorded_time.strftime("%H:%M:%S")
            self.records_tree.insert("", "end", values=(recorded_date or "", recorded_time or "", person_id or "", name or ""))
        today_count = sum(1 for row in rows if row and row[0] and str(row[0])[:10] == date.today().isoformat())
        self.stat_today.configure(text=str(today_count))
        self.records_status.configure(text=f"{len(rows)} total attendance records", fg=COLORS["muted"])

    def open_excel(self):
        if not core.ATTENDANCE_FILE.exists():
            messagebox.showinfo("Excel workbook", "The workbook will be created when you mark the first person present.")
            return
        try:
            os.startfile(core.ATTENDANCE_FILE)
        except OSError as error:
            messagebox.showerror("Could not open Excel", str(error))

    def _set_status(self, text, error):
        color = COLORS["red"] if error else COLORS["forest_light"]
        self.sidebar_status.configure(text=text.upper(), fg=color)

    def close(self):
        if self.capture_state is not None:
            shutil.rmtree(self.capture_state["directory"], ignore_errors=True)
        if self.camera_after_id is not None:
            self.root.after_cancel(self.camera_after_id)
        if self.camera is not None:
            self.camera.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    AttendanceApp().run()