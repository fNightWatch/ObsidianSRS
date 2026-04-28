from __future__ import annotations

import json
import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from srs import (
    COMMANDS_NOTE_NAME,
    DEFAULT_REPORT_LIMIT,
    QUEUE_NOTE_NAME,
    ReportRow,
    SRSStore,
    add_manual_review,
    build_report_rows,
    format_runtime,
    has_pending_commands,
    iter_notes,
    parse_runtime,
    process_command_note,
    resolve_now,
    resolve_vault,
    scan_and_index,
    sync_obsidian_bridge,
    sync_tags,
)

APP_STATE_PATH = Path(__file__).with_name(".obsidiancheck_ui.json")
COMMAND_POLL_INTERVAL_MS = 2500

BAND_FILTERS = {
    "Все уровни": None,
    "Критично": "critical-red",
    "Красное": "red",
    "Оранжевое": "orange",
    "Жёлтое": "yellow",
    "Лайм": "lime",
    "Зелёное": "green",
}

BAND_TITLES = {
    "urgency": "Срочность",
    "band": "Band",
    "reviews": "Повторы",
    "last_diff": "Последняя сложность",
    "last_ts": "Последнее повторение",
    "note": "Заметка",
}

BAND_ROW_COLORS = {
    "critical-red": "#fbe4e7",
    "red": "#fdebdc",
    "orange": "#fff2d9",
    "yellow": "#fff7d6",
    "lime": "#f2fadf",
    "green": "#edf8e9",
}


class SRSDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Obsidian SRS Control Panel")
        self.root.geometry("1260x880")
        self.root.minsize(1120, 780)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.result_queue: queue.Queue[tuple[str, object, object | None]] = queue.Queue()
        self.busy = False
        self.note_paths: list[str] = []
        self.report_rows: list[ReportRow] = []
        self.filtered_rows: list[ReportRow] = []
        self.report_index_by_path: dict[str, ReportRow] = {}
        self.preview_note_path: str | None = None
        self.preview_text_widgets: list[tk.Text] = []
        self.action_widgets: list[ttk.Widget] = []
        self.tabs: dict[str, ttk.Frame] = {}

        self.vault_var = tk.StringVar()
        self.simulation_var = tk.StringVar()
        self.effective_time_var = tk.StringVar(value="Будет использовано текущее время.")
        self.last_scan_var = tk.StringVar(value="Пока не было")
        self.last_sync_var = tk.StringVar(value="Пока не было")
        self.last_action_var = tk.StringVar(value="Нет записей")
        self.last_command_var = tk.StringVar(value="Пока не было")
        self.coverage_var = tk.StringVar(value="0/0")
        self.queue_note_var = tk.StringVar(value=QUEUE_NOTE_NAME)
        self.command_note_var = tk.StringVar(value=COMMANDS_NOTE_NAME)
        self.command_watch_var = tk.StringVar(value="Активно · ожидание команд")
        self.review_note_var = tk.StringVar()
        self.review_difficulty_var = tk.IntVar(value=5)
        self.report_limit_var = tk.IntVar(value=DEFAULT_REPORT_LIMIT)
        self.report_search_var = tk.StringVar()
        self.report_band_var = tk.StringVar(value="Все уровни")
        self.report_count_var = tk.StringVar(value="Пока нет данных")
        self.status_var = tk.StringVar(value="Готово.")

        self.critical_notes_var = tk.StringVar(value="0")
        self.high_priority_var = tk.StringVar(value="0")
        self.avg_difficulty_var = tk.StringVar(value="—")
        self.next_note_var = tk.StringVar(value="—")

        self.preview_title_var = tk.StringVar(value="Выберите заметку")
        self.preview_meta_var = tk.StringVar(value="Здесь появится краткая сводка по выбранной заметке.")

        self.sort_column = "urgency"
        self.sort_descending = True

        self._configure_style()
        self._build_ui()
        self._bind_shortcuts()
        self._load_app_state()

        self.simulation_var.trace_add("write", self._on_simulation_changed)
        self.review_note_var.trace_add("write", self._on_review_note_changed)
        self.report_search_var.trace_add("write", self._on_report_filter_changed)
        self.report_band_var.trace_add("write", self._on_report_filter_changed)
        self.report_limit_var.trace_add("write", self._on_report_filter_changed)

        self._on_simulation_changed()
        self.root.after(100, self._poll_queue)
        self.root.after(1500, self._poll_obsidian_commands)

    def _configure_style(self) -> None:
        self.root.configure(bg="#f5f1ea")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("App.TFrame", background="#f5f1ea")
        style.configure("Surface.TFrame", background="#ffffff")
        style.configure("Card.TFrame", background="#fffaf2")
        style.configure("MutedCard.TFrame", background="#f8f4ec")
        style.configure(
            "Header.TLabel",
            background="#f5f1ea",
            foreground="#2d261f",
            font=("Segoe UI Semibold", 21),
        )
        style.configure(
            "SectionTitle.TLabel",
            background="#ffffff",
            foreground="#2d261f",
            font=("Segoe UI Semibold", 12),
        )
        style.configure(
            "BridgeValue.TLabel",
            background="#f5f1ea",
            foreground="#2d261f",
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Sub.TLabel",
            background="#f5f1ea",
            foreground="#6d655c",
            font=("Segoe UI", 10),
        )
        style.configure(
            "SurfaceSub.TLabel",
            background="#ffffff",
            foreground="#6d655c",
            font=("Segoe UI", 10),
        )
        style.configure(
            "CardLabel.TLabel",
            background="#fffaf2",
            foreground="#7a6f62",
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardValue.TLabel",
            background="#fffaf2",
            foreground="#2d261f",
            font=("Segoe UI Semibold", 12),
        )
        style.configure(
            "PreviewTitle.TLabel",
            background="#ffffff",
            foreground="#2d261f",
            font=("Segoe UI Semibold", 13),
        )
        style.configure("Section.TLabelframe", background="#f5f1ea")
        style.configure(
            "Section.TLabelframe.Label",
            background="#f5f1ea",
            foreground="#2d261f",
            font=("Segoe UI Semibold", 11),
        )
        style.configure("Surface.TLabelframe", background="#ffffff")
        style.configure(
            "Surface.TLabelframe.Label",
            background="#ffffff",
            foreground="#2d261f",
            font=("Segoe UI Semibold", 11),
        )
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=(12, 8))
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(10, 8))
        style.configure("Small.TButton", font=("Segoe UI Semibold", 9), padding=(8, 4))
        style.configure("App.TNotebook", background="#f5f1ea", borderwidth=0)
        style.configure("App.TNotebook.Tab", padding=(16, 8), font=("Segoe UI Semibold", 10))
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", "#fffaf2")],
            foreground=[("selected", "#2d261f")],
        )
        style.configure("Treeview", rowheight=28, font=("Consolas", 10), fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))
        style.configure("TSeparator", background="#e7e0d6")

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, padding=18, style="App.TFrame")
        shell.pack(fill="both", expand=True)

        header = ttk.Frame(shell, style="App.TFrame")
        header.pack(fill="x")

        title_wrap = ttk.Frame(header, style="App.TFrame")
        title_wrap.pack(side="left", fill="x", expand=True)
        ttk.Label(title_wrap, text="Obsidian SRS", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            title_wrap,
            text="Меньше визуального шума, отдельные вкладки под сценарии и быстрый доступ к заметкам, которые реально стоит повторить сейчас.",
            style="Sub.TLabel",
            wraplength=820,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        toolbar = ttk.Frame(shell, style="App.TFrame")
        toolbar.pack(fill="x", pady=(14, 0))

        refresh_button = ttk.Button(toolbar, text="Обновить статус", command=self.refresh_status, style="Primary.TButton")
        refresh_button.pack(side="left")
        scan_button = ttk.Button(toolbar, text="Сканировать маркеры", command=self.run_scan, style="Primary.TButton")
        scan_button.pack(side="left", padx=(8, 0))
        sync_button = ttk.Button(toolbar, text="Синхронизировать теги", command=self.run_sync, style="Primary.TButton")
        sync_button.pack(side="left", padx=(8, 0))
        report_button = ttk.Button(toolbar, text="Обновить отчёт", command=self.run_report, style="Primary.TButton")
        report_button.pack(side="left", padx=(8, 0))
        commands_button = ttk.Button(
            toolbar,
            text="Команды из Obsidian",
            command=self.run_obsidian_commands,
            style="Secondary.TButton",
        )
        commands_button.pack(side="left", padx=(8, 0))
        ttk.Label(
            toolbar,
            text="F5 — статус · Ctrl+R — отчёт · Ctrl+Enter — review · _SRS Commands.md читается автоматически",
            style="Sub.TLabel",
        ).pack(side="right")
        self.action_widgets.extend([refresh_button, scan_button, sync_button, report_button, commands_button])

        self.notebook = ttk.Notebook(shell, style="App.TNotebook")
        self.notebook.pack(fill="both", expand=True, pady=(14, 0))
        self.notebook.enable_traversal()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        overview_tab = ttk.Frame(self.notebook, padding=12, style="App.TFrame")
        review_tab = ttk.Frame(self.notebook, padding=12, style="App.TFrame")
        report_tab = ttk.Frame(self.notebook, padding=12, style="App.TFrame")
        activity_tab = ttk.Frame(self.notebook, padding=12, style="App.TFrame")

        self.notebook.add(overview_tab, text="Обзор")
        self.notebook.add(review_tab, text="Ревью")
        self.notebook.add(report_tab, text="Отчёт")
        self.notebook.add(activity_tab, text="Активность")

        self.tabs = {
            "overview": overview_tab,
            "review": review_tab,
            "report": report_tab,
            "activity": activity_tab,
        }

        self._build_overview_tab(overview_tab)
        self._build_review_tab(review_tab)
        self._build_report_tab(report_tab)
        self._build_activity_tab(activity_tab)

        footer = ttk.Frame(shell, style="App.TFrame")
        footer.pack(fill="x", pady=(12, 0))
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=False)
        self.progress.configure(length=220)
        ttk.Label(footer, textvariable=self.status_var, style="Sub.TLabel").pack(side="left")

    def _build_overview_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(2, weight=1)

        study_cards = ttk.Frame(parent, style="App.TFrame")
        study_cards.grid(row=0, column=0, columnspan=2, sticky="ew")
        for column in range(4):
            study_cards.columnconfigure(column, weight=1)
        self._build_value_card(study_cards, "Критичные сейчас", self.critical_notes_var, 0)
        self._build_value_card(study_cards, "Высокий приоритет", self.high_priority_var, 1)
        self._build_value_card(study_cards, "Средняя сложность", self.avg_difficulty_var, 2)
        self._build_value_card(study_cards, "Следующая заметка", self.next_note_var, 3)

        system_cards = ttk.Frame(parent, style="App.TFrame")
        system_cards.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        for column in range(4):
            system_cards.columnconfigure(column, weight=1)
        self._build_value_card(system_cards, "Последний скан", self.last_scan_var, 0)
        self._build_value_card(system_cards, "Последняя синхронизация", self.last_sync_var, 1)
        self._build_value_card(system_cards, "Последнее действие", self.last_action_var, 2)
        self._build_value_card(system_cards, "Охват индекса", self.coverage_var, 3)

        left_column = ttk.Frame(parent, style="App.TFrame")
        left_column.grid(row=2, column=0, sticky="nsew", pady=(12, 0), padx=(0, 8))
        left_column.columnconfigure(0, weight=1)

        vault_frame = ttk.LabelFrame(left_column, text="Vault", padding=12, style="Section.TLabelframe")
        vault_frame.grid(row=0, column=0, sticky="ew")
        vault_frame.columnconfigure(1, weight=1)
        ttk.Label(vault_frame, text="Папка:", style="Sub.TLabel").grid(row=0, column=0, sticky="w")
        vault_entry = ttk.Entry(vault_frame, textvariable=self.vault_var)
        vault_entry.grid(row=0, column=1, sticky="ew", padx=(8, 10))
        browse_button = ttk.Button(vault_frame, text="Выбрать папку", command=self.choose_vault, style="Secondary.TButton")
        browse_button.grid(row=0, column=2, sticky="ew")
        ttk.Label(
            vault_frame,
            text="Путь сохраняется между запусками. После смены vault статус и список заметок обновляются автоматически.",
            style="Sub.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.action_widgets.extend([vault_entry, browse_button])

        sim_frame = ttk.LabelFrame(left_column, text="Время выполнения", padding=12, style="Section.TLabelframe")
        sim_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        sim_frame.columnconfigure(1, weight=1)
        ttk.Label(sim_frame, text="Симуляция:", style="Sub.TLabel").grid(row=0, column=0, sticky="w")
        sim_entry = ttk.Entry(sim_frame, textvariable=self.simulation_var)
        sim_entry.grid(row=0, column=1, sticky="ew", padx=(8, 10))
        stamp_button = ttk.Button(sim_frame, text="Подставить сейчас", command=self.fill_current_time, style="Secondary.TButton")
        stamp_button.grid(row=0, column=2, sticky="ew")
        clear_button = ttk.Button(sim_frame, text="Очистить", command=self.clear_simulation_time, style="Secondary.TButton")
        clear_button.grid(row=0, column=3, sticky="ew", padx=(8, 0))
        ttk.Label(
            sim_frame,
            text="Формат: 2026-04-01 09:30 или ISO со смещением. Пустое поле = реальное текущее время.",
            style="Sub.TLabel",
            wraplength=560,
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(sim_frame, textvariable=self.effective_time_var, style="Sub.TLabel").grid(
            row=2,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(6, 0),
        )
        self.action_widgets.extend([sim_entry, stamp_button, clear_button])

        workflow_frame = ttk.LabelFrame(left_column, text="Как это ускоряет учебу", padding=12, style="Section.TLabelframe")
        workflow_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        workflow_lines = [
            "1. Начинайте с вкладки «Обзор»: она показывает, что повторять прямо сейчас.",
            "2. Двойной клик по заметке переносит её в «Ревью» без лишнего поиска.",
            "3. На вкладке «Отчёт» можно быстро отфильтровать очередь по band и по имени заметки.",
            "4. Предпросмотр заметки снижает количество лишних переключений обратно в Obsidian.",
        ]
        for idx, line in enumerate(workflow_lines):
            ttk.Label(
                workflow_frame,
                text=line,
                style="Sub.TLabel",
                wraplength=560,
                justify="left",
            ).grid(row=idx, column=0, sticky="w", pady=(0 if idx == 0 else 6, 0))

        right_column = ttk.Frame(parent, style="App.TFrame")
        right_column.grid(row=2, column=1, sticky="nsew", pady=(12, 0), padx=(8, 0))
        right_column.columnconfigure(0, weight=1)
        right_column.rowconfigure(0, weight=3)
        right_column.rowconfigure(1, weight=2)

        focus_frame = ttk.LabelFrame(right_column, text="Сегодня в фокусе", padding=12, style="Section.TLabelframe")
        focus_frame.grid(row=0, column=0, sticky="nsew")
        focus_frame.columnconfigure(0, weight=1)
        focus_frame.rowconfigure(1, weight=1)
        ttk.Label(
            focus_frame,
            text="Топ заметок по срочности. Двойной клик переносит заметку в ревью.",
            style="Sub.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        focus_table = ttk.Frame(focus_frame)
        focus_table.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        focus_table.columnconfigure(0, weight=1)
        focus_table.rowconfigure(0, weight=1)
        self.focus_tree = ttk.Treeview(focus_table, columns=("urgency", "note"), show="headings", height=10)
        self.focus_tree.grid(row=0, column=0, sticky="nsew")
        self.focus_tree.heading("urgency", text="Срочность")
        self.focus_tree.heading("note", text="Заметка")
        self.focus_tree.column("urgency", width=90, stretch=False, anchor="w")
        self.focus_tree.column("note", width=380, anchor="w")
        focus_scroll = ttk.Scrollbar(focus_table, orient="vertical", command=self.focus_tree.yview)
        focus_scroll.grid(row=0, column=1, sticky="ns")
        self.focus_tree.configure(yscrollcommand=focus_scroll.set)
        self.focus_tree.bind("<<TreeviewSelect>>", self._on_focus_selected)
        self.focus_tree.bind("<Double-1>", self._on_focus_open_for_review)

        focus_actions = ttk.Frame(focus_frame, style="App.TFrame")
        focus_actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        next_button = ttk.Button(
            focus_actions,
            text="Подставить следующую в ревью",
            command=self.pick_next_focus_note,
            style="Primary.TButton",
        )
        next_button.pack(side="left")
        report_tab_button = ttk.Button(
            focus_actions,
            text="Открыть отчёт",
            command=lambda: self._select_tab("report"),
            style="Secondary.TButton",
        )
        report_tab_button.pack(side="left", padx=(8, 0))
        self.action_widgets.extend([next_button, report_tab_button])

        bridge_frame = ttk.LabelFrame(right_column, text="Obsidian bridge", padding=12, style="Section.TLabelframe")
        bridge_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        bridge_frame.columnconfigure(1, weight=1)
        ttk.Label(
            bridge_frame,
            text="Очередь и команды живут прямо во vault, чтобы основной рабочий цикл оставался внутри Obsidian.",
            style="Sub.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(bridge_frame, text="Очередь:", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Label(bridge_frame, textvariable=self.queue_note_var, style="BridgeValue.TLabel").grid(
            row=1,
            column=1,
            sticky="w",
            pady=(10, 0),
        )
        ttk.Label(bridge_frame, text="Команды:", style="Sub.TLabel").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(bridge_frame, textvariable=self.command_note_var, style="BridgeValue.TLabel").grid(
            row=2,
            column=1,
            sticky="w",
            pady=(6, 0),
        )
        ttk.Label(bridge_frame, text="Последняя команда:", style="Sub.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Label(bridge_frame, textvariable=self.last_command_var, style="BridgeValue.TLabel").grid(
            row=3,
            column=1,
            sticky="w",
            pady=(6, 0),
        )
        ttk.Label(bridge_frame, text="Режим слежения:", style="Sub.TLabel").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Label(bridge_frame, textvariable=self.command_watch_var, style="BridgeValue.TLabel").grid(
            row=4,
            column=1,
            sticky="w",
            pady=(6, 0),
        )

        bridge_actions = ttk.Frame(bridge_frame, style="App.TFrame")
        bridge_actions.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        process_commands_button = ttk.Button(
            bridge_actions,
            text="Проверить команды сейчас",
            command=self.run_obsidian_commands,
            style="Primary.TButton",
        )
        process_commands_button.pack(side="left")
        refresh_bridge_button = ttk.Button(
            bridge_actions,
            text="Перестроить md-очередь",
            command=self.run_obsidian_bridge_refresh,
            style="Secondary.TButton",
        )
        refresh_bridge_button.pack(side="left", padx=(8, 0))
        self.action_widgets.extend([process_commands_button, refresh_bridge_button])

    def _build_review_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=2)
        parent.columnconfigure(1, weight=3)
        parent.rowconfigure(0, weight=1)

        left = ttk.Frame(parent, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        review_frame = ttk.LabelFrame(left, text="Ручное ревью", padding=12, style="Section.TLabelframe")
        review_frame.grid(row=0, column=0, sticky="ew")
        review_frame.columnconfigure(1, weight=1)

        ttk.Label(review_frame, text="Заметка:", style="Sub.TLabel").grid(row=0, column=0, sticky="w")
        self.note_combo = ttk.Combobox(review_frame, textvariable=self.review_note_var)
        self.note_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 10))
        ttk.Button(
            review_frame,
            text="Следующая из фокуса",
            command=self.pick_next_focus_note,
            style="Secondary.TButton",
        ).grid(row=0, column=4, sticky="ew")

        ttk.Label(review_frame, text="Сложность:", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        difficulty_spin = ttk.Spinbox(review_frame, from_=1, to=10, textvariable=self.review_difficulty_var, width=6)
        difficulty_spin.grid(row=1, column=1, sticky="w", padx=(8, 10), pady=(10, 0))

        presets = ttk.Frame(review_frame, style="App.TFrame")
        presets.grid(row=1, column=2, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(presets, text="Быстрые значения:", style="Sub.TLabel").pack(side="left")
        for value in (2, 4, 6, 8, 10):
            ttk.Button(
                presets,
                text=str(value),
                command=lambda preset=value: self._set_quick_difficulty(preset),
                style="Small.TButton",
                width=3,
            ).pack(side="left", padx=(6, 0))

        review_button = ttk.Button(
            review_frame,
            text="Добавить review",
            command=self.run_manual_review,
            style="Primary.TButton",
        )
        review_button.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(12, 0))
        ttk.Label(
            review_frame,
            text="Двойной клик по заметке в «Обзоре» или «Отчёте» автоматически переносит её сюда.",
            style="Sub.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))
        self.action_widgets.extend([self.note_combo, difficulty_spin, review_button])

        quick_help = ttk.LabelFrame(left, text="Быстрый ритм работы", padding=12, style="Section.TLabelframe")
        quick_help.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        help_lines = [
            "• Возьмите верхнюю заметку из фокуса, не тратьте время на выбор вручную.",
            "• Используйте быстрые значения сложности вместо ручного ввода.",
            "• После review сразу обновляйте отчёт, чтобы видеть новую очередь приоритета.",
        ]
        for idx, line in enumerate(help_lines):
            ttk.Label(quick_help, text=line, style="Sub.TLabel", wraplength=520, justify="left").grid(
                row=idx,
                column=0,
                sticky="w",
                pady=(0 if idx == 0 else 6, 0),
            )

        right = ttk.Frame(parent, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(right, text="Предпросмотр заметки", padding=12, style="Surface.TLabelframe")
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(2, weight=1)

        ttk.Label(preview_frame, textvariable=self.preview_title_var, style="PreviewTitle.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            preview_frame,
            textvariable=self.preview_meta_var,
            style="SurfaceSub.TLabel",
            wraplength=700,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 10))

        preview_text = self._create_readonly_text(preview_frame, row=2, column=0, bg="#fffdf8")
        self.preview_text_widgets.append(preview_text)

        review_actions = ttk.Frame(preview_frame, style="Surface.TFrame")
        review_actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        copy_path_button = ttk.Button(
            review_actions,
            text="Скопировать путь",
            command=self._copy_selected_note_path,
            style="Secondary.TButton",
        )
        copy_path_button.pack(side="left")
        report_jump_button = ttk.Button(
            review_actions,
            text="Открыть в отчёте",
            command=lambda: self._select_tab("report"),
            style="Secondary.TButton",
        )
        report_jump_button.pack(side="left", padx=(8, 0))
        self.action_widgets.extend([copy_path_button, report_jump_button])

    def _build_report_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="Фильтры и управление", padding=12, style="Section.TLabelframe")
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=0)

        ttk.Label(controls, text="Поиск:", style="Sub.TLabel").grid(row=0, column=0, sticky="w")
        self.report_search_entry = ttk.Entry(controls, textvariable=self.report_search_var)
        self.report_search_entry.grid(row=0, column=1, sticky="ew", padx=(8, 12))
        ttk.Label(controls, text="Band:", style="Sub.TLabel").grid(row=0, column=2, sticky="w")
        self.report_band_combo = ttk.Combobox(
            controls,
            textvariable=self.report_band_var,
            values=list(BAND_FILTERS.keys()),
            state="readonly",
            width=18,
        )
        self.report_band_combo.grid(row=0, column=3, sticky="ew", padx=(8, 12))
        ttk.Label(controls, text="Лимит:", style="Sub.TLabel").grid(row=0, column=4, sticky="w")
        limit_spin = ttk.Spinbox(controls, from_=5, to=1000, textvariable=self.report_limit_var, width=6)
        limit_spin.grid(row=0, column=5, sticky="w", padx=(8, 12))
        clear_filters = ttk.Button(controls, text="Сбросить фильтры", command=self._clear_report_filters, style="Secondary.TButton")
        clear_filters.grid(row=0, column=6, sticky="ew")
        rebuild_button = ttk.Button(controls, text="Перестроить", command=self.run_report, style="Primary.TButton")
        rebuild_button.grid(row=0, column=7, sticky="ew", padx=(8, 0))
        ttk.Label(controls, textvariable=self.report_count_var, style="Sub.TLabel").grid(
            row=1,
            column=0,
            columnspan=8,
            sticky="w",
            pady=(8, 0),
        )
        self.action_widgets.extend([
            self.report_search_entry,
            self.report_band_combo,
            limit_spin,
            clear_filters,
            rebuild_button,
        ])

        content = ttk.PanedWindow(parent, orient="horizontal")
        content.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        table_panel = ttk.Frame(content, style="App.TFrame")
        preview_panel = ttk.Frame(content, style="App.TFrame")
        content.add(table_panel, weight=4)
        content.add(preview_panel, weight=3)

        table_panel.columnconfigure(0, weight=1)
        table_panel.rowconfigure(0, weight=1)
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(0, weight=1)

        table_frame = ttk.LabelFrame(table_panel, text="Очередь по приоритету", padding=12, style="Section.TLabelframe")
        table_frame.grid(row=0, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        tree_wrap = ttk.Frame(table_frame)
        tree_wrap.grid(row=0, column=0, sticky="nsew")
        tree_wrap.columnconfigure(0, weight=1)
        tree_wrap.rowconfigure(0, weight=1)

        columns = ("urgency", "band", "reviews", "last_diff", "last_ts", "note")
        self.report_tree = ttk.Treeview(tree_wrap, columns=columns, show="headings")
        self.report_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.report_tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(tree_wrap, orient="horizontal", command=self.report_tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.report_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        column_widths = {
            "urgency": 90,
            "band": 110,
            "reviews": 90,
            "last_diff": 150,
            "last_ts": 190,
            "note": 500,
        }
        for column_id, title in BAND_TITLES.items():
            self.report_tree.heading(column_id, text=title, command=lambda c=column_id: self._on_report_heading_clicked(c))
            self.report_tree.column(column_id, width=column_widths[column_id], anchor="w")
        for tag, color in BAND_ROW_COLORS.items():
            self.report_tree.tag_configure(tag, background=color)

        self.report_tree.bind("<<TreeviewSelect>>", self._on_report_selected)
        self.report_tree.bind("<Double-1>", self._on_report_open_for_review)

        preview_frame = ttk.LabelFrame(preview_panel, text="Выбранная заметка", padding=12, style="Surface.TLabelframe")
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(2, weight=1)

        ttk.Label(preview_frame, textvariable=self.preview_title_var, style="PreviewTitle.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            preview_frame,
            textvariable=self.preview_meta_var,
            style="SurfaceSub.TLabel",
            wraplength=560,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 10))

        report_preview_text = self._create_readonly_text(preview_frame, row=2, column=0, bg="#ffffff")
        self.preview_text_widgets.append(report_preview_text)

        report_actions = ttk.Frame(preview_frame, style="Surface.TFrame")
        report_actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        send_to_review = ttk.Button(
            report_actions,
            text="Перенести в ревью",
            command=self._open_selected_report_note_for_review,
            style="Primary.TButton",
        )
        send_to_review.pack(side="left")
        copy_report_path = ttk.Button(
            report_actions,
            text="Скопировать путь",
            command=self._copy_selected_note_path,
            style="Secondary.TButton",
        )
        copy_report_path.pack(side="left", padx=(8, 0))
        self.action_widgets.extend([send_to_review, copy_report_path])

    def _build_activity_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)

        log_frame = ttk.LabelFrame(parent, text="Лог", padding=12, style="Section.TLabelframe")
        log_frame.grid(row=0, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        log_wrap = ttk.Frame(log_frame)
        log_wrap.grid(row=0, column=0, sticky="nsew")
        log_wrap.columnconfigure(0, weight=1)
        log_wrap.rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_wrap,
            height=16,
            wrap="word",
            bg="#1f1a17",
            fg="#f9f2e7",
            insertbackground="#f9f2e7",
            relief="flat",
            font=("Consolas", 10),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set, state="disabled")

        tips_frame = ttk.LabelFrame(parent, text="Горячие клавиши и рабочий поток", padding=12, style="Section.TLabelframe")
        tips_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        tips = [
            "F5 — обновить статус и сводку по заметкам.",
            "Ctrl+R — перестроить отчёт без кликов мышью.",
            "Ctrl+Shift+S — быстро просканировать маркеры @! и занести review в индекс.",
            "Ctrl+Shift+Y — пересчитать tag band после серии review.",
            "Ctrl+Enter — добавить review для текущей заметки на вкладке «Ревью».",
            "Команды можно писать прямо в _SRS Commands.md: приложение автоматически подхватит их и очистит секцию Inbox.",
        ]
        for idx, line in enumerate(tips):
            ttk.Label(tips_frame, text=line, style="Sub.TLabel", wraplength=1160, justify="left").grid(
                row=idx,
                column=0,
                sticky="w",
                pady=(0 if idx == 0 else 6, 0),
            )

        clear_log_button = ttk.Button(tips_frame, text="Очистить лог", command=self._clear_log, style="Secondary.TButton")
        clear_log_button.grid(row=len(tips), column=0, sticky="w", pady=(10, 0))
        self.action_widgets.append(clear_log_button)

    def _build_value_card(self, parent: ttk.Frame, title: str, variable: tk.StringVar, column: int) -> None:
        card = ttk.Frame(parent, padding=12, style="Card.TFrame")
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 10, 0))
        ttk.Label(card, text=title, style="CardLabel.TLabel").pack(anchor="w")
        ttk.Label(
            card,
            textvariable=variable,
            style="CardValue.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

    def _create_readonly_text(self, parent: ttk.Frame, row: int, column: int, *, bg: str) -> tk.Text:
        wrap = ttk.Frame(parent)
        wrap.grid(row=row, column=column, sticky="nsew")
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        text = tk.Text(
            wrap,
            wrap="word",
            relief="flat",
            bg=bg,
            fg="#1f1a17",
            insertbackground="#1f1a17",
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(wrap, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set, state="disabled")
        return text

    def _bind_shortcuts(self) -> None:
        self.root.bind("<F5>", lambda _event: self.refresh_status())
        self.root.bind("<Control-r>", self._on_ctrl_report)
        self.root.bind("<Control-Shift-S>", self._on_ctrl_scan)
        self.root.bind("<Control-Shift-Y>", self._on_ctrl_sync)
        self.root.bind("<Control-Return>", self._on_ctrl_review)

    def _on_ctrl_report(self, _event: tk.Event[tk.Misc]) -> str:
        self.run_report()
        return "break"

    def _on_ctrl_scan(self, _event: tk.Event[tk.Misc]) -> str:
        self.run_scan()
        return "break"

    def _on_ctrl_sync(self, _event: tk.Event[tk.Misc]) -> str:
        self.run_sync()
        return "break"

    def _on_ctrl_review(self, _event: tk.Event[tk.Misc]) -> str:
        self.run_manual_review()
        return "break"

    def _on_close(self) -> None:
        self._save_app_state()
        self.root.destroy()

    def _on_tab_changed(self, _event: tk.Event[tk.Misc]) -> None:
        self._save_app_state()

    def _load_app_state(self) -> None:
        if not APP_STATE_PATH.exists():
            return
        try:
            payload = json.loads(APP_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        geometry = payload.get("geometry")
        if isinstance(geometry, str) and geometry:
            try:
                self.root.geometry(geometry)
            except tk.TclError:
                pass

        last_vault = payload.get("last_vault")
        if isinstance(last_vault, str) and last_vault:
            self.vault_var.set(last_vault)
            self.refresh_status(show_errors=False, add_log=False)

        selected_tab = payload.get("selected_tab")
        if isinstance(selected_tab, str) and selected_tab in self.tabs:
            self._select_tab(selected_tab)

    def _save_app_state(self) -> None:
        payload = {
            "last_vault": self.vault_var.get().strip(),
            "selected_tab": self._current_tab_key(),
            "geometry": self.root.geometry(),
        }
        try:
            APP_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _current_tab_key(self) -> str:
        selected = self.notebook.select()
        for key, frame in self.tabs.items():
            if str(frame) == selected:
                return key
        return "overview"

    def _select_tab(self, key: str) -> None:
        frame = self.tabs.get(key)
        if frame is None:
            return
        self.notebook.select(frame)

    def _on_simulation_changed(self, *_args: object) -> None:
        raw = self.simulation_var.get().strip()
        if not raw:
            self.effective_time_var.set("Будет использовано текущее время.")
            return
        try:
            simulated = parse_runtime(raw)
        except Exception:
            self.effective_time_var.set("Неверный формат даты. Используйте ISO-подобный формат.")
            return
        self.effective_time_var.set(f"Будет использовано: {simulated.astimezone().strftime('%Y-%m-%d %H:%M:%S')}")

    def _on_review_note_changed(self, *_args: object) -> None:
        note_path = self.review_note_var.get().strip()
        if not note_path:
            return
        self._highlight_note_in_lists(note_path)
        self._show_note_preview(note_path)

    def _on_report_filter_changed(self, *_args: object) -> None:
        self._apply_report_filters()

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.status_var.set("Лог очищен.")

    def _set_busy(self, busy: bool, status_text: str | None = None) -> None:
        self.busy = busy
        for widget in self.action_widgets:
            if busy:
                widget.state(["disabled"])
            else:
                widget.state(["!disabled"])

        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()
        self.status_var.set(status_text or ("Выполняется..." if busy else "Готово."))

    def _safe_int(self, value: object, default: int, *, minimum: int = 1, maximum: int = 10_000) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError, tk.TclError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def choose_vault(self) -> None:
        initial_dir = self.vault_var.get().strip() or str(Path.home())
        selected = filedialog.askdirectory(title="Выберите Obsidian vault", initialdir=initial_dir)
        if not selected:
            return
        self.vault_var.set(selected)
        self.refresh_status()

    def refresh_status(self, *, show_errors: bool = True, add_log: bool = True) -> None:
        try:
            vault = self._require_vault()
            run_at = self._current_runtime()
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Vault", str(exc))
            return

        self._refresh_status_from_vault(vault)
        store = SRSStore(vault)
        rows = self._build_full_report(vault, store, run_at)
        synced_rows, queue_summary = sync_obsidian_bridge(vault, store, run_at, rows=rows)
        self._set_report_rows(synced_rows)
        self._refresh_status_from_vault(vault)

        if add_log:
            self._append_log(
                f"Статус обновлён: {vault} · очередь: {queue_summary.path} ({queue_summary.row_count} заметок)"
            )
        self.status_var.set("Статус обновлён.")

    def fill_current_time(self) -> None:
        self.simulation_var.set(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"))

    def clear_simulation_time(self) -> None:
        self.simulation_var.set("")

    def run_obsidian_bridge_refresh(self) -> None:
        def worker(vault: Path, run_at: datetime, emit: Callable[[str], None]) -> dict[str, object]:
            store = SRSStore(vault)
            rows = self._build_full_report(vault, store, run_at)
            synced_rows, queue_summary = sync_obsidian_bridge(vault, store, run_at, rows=rows, emit=emit)
            return {
                "rows": synced_rows,
                "queue_summary": {
                    "path": queue_summary.path,
                    "row_count": queue_summary.row_count,
                    "untracked_count": queue_summary.untracked_count,
                },
            }

        self._run_background_action("bridge", "Обновление файлов Obsidian", worker)

    def run_obsidian_commands(self, *, auto: bool = False) -> None:
        if self.busy:
            return

        raw_vault = self.vault_var.get().strip()
        if not raw_vault:
            if not auto:
                messagebox.showerror("Obsidian", "Сначала выберите папку с Obsidian vault.")
            return

        try:
            vault = resolve_vault(raw_vault)
        except Exception as exc:
            if not auto:
                messagebox.showerror("Obsidian", str(exc))
            return

        if auto:
            try:
                if not has_pending_commands(vault):
                    return
            except Exception:
                return

        def worker(vault: Path, run_at: datetime, emit: Callable[[str], None]) -> dict[str, object]:
            store = SRSStore(vault)
            commands, rows, queue_summary = process_command_note(vault, store, run_at, emit=emit)
            return {
                "rows": rows,
                "commands": [item.to_payload() for item in commands],
                "queue_summary": {
                    "path": queue_summary.path,
                    "row_count": queue_summary.row_count,
                    "untracked_count": queue_summary.untracked_count,
                },
            }

        self._run_background_action("commands", "Обработка команд из Obsidian", worker)

    def _poll_obsidian_commands(self) -> None:
        try:
            if not self.busy:
                self.run_obsidian_commands(auto=True)
        except Exception:
            pass
        self.root.after(COMMAND_POLL_INTERVAL_MS, self._poll_obsidian_commands)

    def pick_next_focus_note(self) -> None:
        if not self.report_rows:
            messagebox.showinfo("Ревью", "Пока нет заметок с review-историей для очереди приоритета.")
            return
        self._send_note_to_review(self.report_rows[0].rel_path, switch_tab=True)

    def _require_vault(self) -> Path:
        raw = self.vault_var.get().strip()
        if not raw:
            raise ValueError("Сначала выберите папку с Obsidian vault.")
        vault = resolve_vault(raw)
        normalized = str(vault)
        if normalized != raw:
            self.vault_var.set(normalized)
        self._save_app_state()
        return vault

    def _refresh_status_from_vault(self, vault: Path) -> None:
        store = SRSStore(vault)
        self.note_paths = [note.relative_to(vault).as_posix() for note in iter_notes(vault)]
        self.note_paths.sort()
        self.note_combo.configure(values=self.note_paths)

        self.last_scan_var.set(format_runtime(store.get_meta_timestamp("last_scan_ts"), fallback="Пока не было"))
        self.last_sync_var.set(format_runtime(store.get_meta_timestamp("last_sync_ts"), fallback="Пока не было"))
        self.last_action_var.set(format_runtime(store.get_meta_timestamp("last_action_ts"), fallback="Нет записей"))
        self.last_command_var.set(format_runtime(store.get_meta_timestamp("last_command_ts"), fallback="Пока не было"))
        self.coverage_var.set(f"{store.tracked_notes()}/{len(self.note_paths)}")
        self.queue_note_var.set(QUEUE_NOTE_NAME)
        self.command_note_var.set(COMMANDS_NOTE_NAME)
        try:
            pending_commands = has_pending_commands(vault)
        except Exception:
            pending_commands = False
        self.command_watch_var.set(
            "Есть команды к выполнению" if pending_commands else "Активно · ожидание команд"
        )

        current_note = self.review_note_var.get().strip()
        if current_note and current_note in self.note_paths:
            return
        if self.report_rows:
            self.review_note_var.set(self.report_rows[0].rel_path)
        elif self.note_paths:
            self.review_note_var.set(self.note_paths[0])
        else:
            self.review_note_var.set("")
            self._show_note_preview(None)

    def _current_runtime(self) -> datetime:
        raw = self.simulation_var.get().strip()
        if not raw:
            return resolve_now()
        return parse_runtime(raw)

    def _build_full_report(self, vault: Path, store: SRSStore, run_at: datetime) -> list[ReportRow]:
        return build_report_rows(vault, store, run_at, limit=None)

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload, extra = self.result_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                    continue
                if kind == "error":
                    self._set_busy(False, "Ошибка.")
                    details = str(extra or "")
                    self._append_log(details)
                    last_line = details.strip().splitlines()[-1] if details.strip() else "Unknown error"
                    messagebox.showerror("Ошибка", f"{payload} завершилось с ошибкой:\n{last_line}")
                    continue
                if kind == "done":
                    action = str(payload)
                    result = extra if isinstance(extra, dict) else {}
                    self._handle_action_done(action, result)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _run_background_action(
        self,
        action: str,
        status_text: str,
        worker: Callable[[Path, datetime, Callable[[str], None]], dict[str, object]],
    ) -> None:
        if self.busy:
            return

        try:
            vault = self._require_vault()
            run_at = self._current_runtime()
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        self._set_busy(True, status_text)
        self._append_log(f"{status_text} @ {run_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}")

        def emit(message: str) -> None:
            self.result_queue.put(("log", message, None))

        def job() -> None:
            try:
                result = worker(vault, run_at, emit)
                self.result_queue.put(("done", action, result))
            except Exception:
                self.result_queue.put(("error", action, traceback.format_exc()))

        threading.Thread(target=job, daemon=True).start()

    def _handle_action_done(self, action: str, result: dict[str, object]) -> None:
        try:
            vault = resolve_vault(self.vault_var.get().strip())
            self._refresh_status_from_vault(vault)
        except Exception:
            vault = None

        rows = result.get("rows")
        if isinstance(rows, list):
            self._set_report_rows([row for row in rows if isinstance(row, ReportRow)])
        elif vault is not None:
            store = SRSStore(vault)
            self._set_report_rows(self._build_full_report(vault, store, self._current_runtime()))

        queue_summary = result.get("queue_summary")
        if isinstance(queue_summary, dict):
            queue_path = str(queue_summary.get("path", QUEUE_NOTE_NAME))
            row_count = int(queue_summary.get("row_count", len(self.report_rows)))
            self._append_log(f"Obsidian-файлы обновлены: {queue_path} ({row_count} заметок)")

        if action == "scan":
            self._append_log(f"Сканирование завершено. Обработано файлов: {result.get('count', 0)}")
        elif action == "sync":
            self._append_log(f"Синхронизация завершена. Обновлено файлов: {result.get('count', 0)}")
        elif action == "review":
            self._append_log("Ручное ревью добавлено.")
        elif action == "report":
            self._append_log(f"Отчёт обновлён. Показано строк: {len(self.filtered_rows)}")
        elif action == "bridge":
            self._append_log("Markdown-очередь внутри Obsidian перестроена.")
        elif action == "commands":
            commands = result.get("commands")
            if isinstance(commands, list) and commands:
                success_count = sum(
                    1 for item in commands if isinstance(item, dict) and item.get("status") == "ok"
                )
                self._append_log(
                    f"Команды из Obsidian обработаны: {success_count}/{len(commands)} успешно."
                )
                for item in commands[:8]:
                    if not isinstance(item, dict):
                        continue
                    icon = "✓" if item.get("status") == "ok" else "!"
                    command = str(item.get("command", "")).strip()
                    message = str(item.get("message", "")).strip()
                    suffix = f" — {message}" if message else ""
                    self._append_log(f"{icon} {command}{suffix}")
            else:
                self._append_log("Команд в файле Obsidian не было: очередь просто освежена.")

        if vault is not None:
            self._refresh_status_from_vault(vault)
        self._set_busy(False, "Готово.")

    def _set_report_rows(self, rows: list[ReportRow]) -> None:
        unique_rows: dict[str, ReportRow] = {row.rel_path: row for row in rows}
        self.report_rows = sorted(unique_rows.values(), key=lambda row: row.urgency, reverse=True)
        self.report_index_by_path = {row.rel_path: row for row in self.report_rows}
        self._update_summary_metrics()
        self._populate_focus_tree(self.report_rows[:8])
        self._apply_report_filters()

        current_note = self.review_note_var.get().strip()
        if current_note and current_note in self.report_index_by_path:
            self._show_note_preview(current_note)
            return
        if current_note and current_note in self.note_paths:
            self._show_note_preview(current_note)
            return
        if self.report_rows:
            self.review_note_var.set(self.report_rows[0].rel_path)
        elif self.note_paths:
            self.review_note_var.set(self.note_paths[0])
        else:
            self._show_note_preview(None)

    def _update_summary_metrics(self) -> None:
        critical_count = sum(1 for row in self.report_rows if row.urgency >= 0.92)
        high_priority_count = sum(1 for row in self.report_rows if row.urgency >= 0.65)
        self.critical_notes_var.set(str(critical_count))
        self.high_priority_var.set(str(high_priority_count))
        if self.report_rows:
            avg_difficulty = sum(row.last_difficulty for row in self.report_rows) / len(self.report_rows)
            self.avg_difficulty_var.set(f"{avg_difficulty:.1f}/10")
            self.next_note_var.set(self.report_rows[0].rel_path)
        else:
            self.avg_difficulty_var.set("—")
            self.next_note_var.set("—")

    def _populate_focus_tree(self, rows: list[ReportRow]) -> None:
        current_selection = self._selected_tree_path(self.focus_tree)
        self.focus_tree.delete(*self.focus_tree.get_children())
        for row in rows:
            self.focus_tree.insert(
                "",
                "end",
                iid=row.rel_path,
                values=(f"{row.urgency:.2f}", row.rel_path),
                tags=(row.band,),
            )
        if current_selection and self.focus_tree.exists(current_selection):
            self.focus_tree.selection_set(current_selection)
            self.focus_tree.see(current_selection)

    def _apply_report_filters(self) -> None:
        if not hasattr(self, "report_tree"):
            return

        rows = list(self.report_rows)
        query = self.report_search_var.get().strip().lower()
        band_filter = BAND_FILTERS.get(self.report_band_var.get())
        limit = self._safe_int(self.report_limit_var.get(), DEFAULT_REPORT_LIMIT, minimum=1, maximum=5000)

        if band_filter:
            rows = [row for row in rows if row.band == band_filter]
        if query:
            rows = [row for row in rows if query in row.rel_path.lower()]

        rows = sorted(rows, key=self._sort_key_for_column, reverse=self.sort_descending)
        self.filtered_rows = rows[:limit]
        self._populate_report(self.filtered_rows)
        self.report_count_var.set(f"Показано {len(self.filtered_rows)} из {len(rows)} заметок")

    def _sort_key_for_column(self, row: ReportRow) -> object:
        if self.sort_column == "urgency":
            return row.urgency
        if self.sort_column == "band":
            return row.band
        if self.sort_column == "reviews":
            return row.review_count
        if self.sort_column == "last_diff":
            return row.last_difficulty
        if self.sort_column == "last_ts":
            return row.last_ts
        return row.rel_path.lower()

    def _on_report_heading_clicked(self, column: str) -> None:
        if self.sort_column == column:
            self.sort_descending = not self.sort_descending
        else:
            self.sort_column = column
            self.sort_descending = column != "note"
        self._update_report_headings()
        self._apply_report_filters()

    def _update_report_headings(self) -> None:
        for column_id, title in BAND_TITLES.items():
            suffix = ""
            if column_id == self.sort_column:
                suffix = " ▼" if self.sort_descending else " ▲"
            self.report_tree.heading(column_id, text=title + suffix, command=lambda c=column_id: self._on_report_heading_clicked(c))

    def _populate_report(self, rows: list[ReportRow]) -> None:
        selected = self._selected_tree_path(self.report_tree)
        self.report_tree.delete(*self.report_tree.get_children())
        for row in rows:
            self.report_tree.insert(
                "",
                "end",
                iid=row.rel_path,
                values=(
                    f"{row.urgency:.2f}",
                    row.band,
                    row.review_count,
                    row.last_difficulty,
                    format_runtime(row.last_ts, fallback=""),
                    row.rel_path,
                ),
                tags=(row.band,),
            )

        if selected and self.report_tree.exists(selected):
            self.report_tree.selection_set(selected)
            self.report_tree.see(selected)
        elif self.filtered_rows:
            first = self.filtered_rows[0].rel_path
            if self.report_tree.exists(first):
                self.report_tree.selection_set(first)
                self.report_tree.see(first)

    def _selected_tree_path(self, tree: ttk.Treeview) -> str | None:
        selected = tree.selection()
        if not selected:
            return None
        return str(selected[0])

    def _on_focus_selected(self, _event: tk.Event[tk.Misc]) -> None:
        note_path = self._selected_tree_path(self.focus_tree)
        if note_path:
            self._show_note_preview(note_path)

    def _on_focus_open_for_review(self, _event: tk.Event[tk.Misc]) -> None:
        note_path = self._selected_tree_path(self.focus_tree)
        if note_path:
            self._send_note_to_review(note_path, switch_tab=True)

    def _on_report_selected(self, _event: tk.Event[tk.Misc]) -> None:
        note_path = self._selected_tree_path(self.report_tree)
        if note_path:
            self._show_note_preview(note_path)

    def _on_report_open_for_review(self, _event: tk.Event[tk.Misc]) -> None:
        self._open_selected_report_note_for_review()

    def _open_selected_report_note_for_review(self) -> None:
        note_path = self._selected_tree_path(self.report_tree)
        if not note_path:
            return
        self._send_note_to_review(note_path, switch_tab=True)

    def _send_note_to_review(self, note_path: str, *, switch_tab: bool) -> None:
        self.review_note_var.set(note_path)
        self._set_quick_difficulty(self._safe_int(self.review_difficulty_var.get(), 5, minimum=1, maximum=10))
        if switch_tab:
            self._select_tab("review")
        self.status_var.set(f"Заметка «{note_path}» перенесена в ревью.")

    def _highlight_note_in_lists(self, note_path: str) -> None:
        if hasattr(self, "focus_tree") and self.focus_tree.exists(note_path):
            self.focus_tree.selection_set(note_path)
            self.focus_tree.see(note_path)
        if hasattr(self, "report_tree") and self.report_tree.exists(note_path):
            self.report_tree.selection_set(note_path)
            self.report_tree.see(note_path)

    def _set_quick_difficulty(self, value: int) -> None:
        self.review_difficulty_var.set(self._safe_int(value, 5, minimum=1, maximum=10))

    def _show_note_preview(self, note_path: str | None) -> None:
        self.preview_note_path = note_path
        if not note_path:
            self.preview_title_var.set("Выберите заметку")
            self.preview_meta_var.set("Здесь появится краткая сводка по выбранной заметке.")
            self._write_preview_text("Нет выбранной заметки.")
            return

        self.preview_title_var.set(note_path)
        row = self.report_index_by_path.get(note_path)
        self.preview_meta_var.set(self._build_preview_meta(row))

        try:
            vault = self._require_vault()
            note = vault / note_path
            if not note.exists():
                self._write_preview_text("Файл заметки не найден во vault.")
                return
            content = note.read_text(encoding="utf-8")
        except Exception as exc:
            self._write_preview_text(f"Не удалось прочитать заметку: {exc}")
            return

        self._write_preview_text(self._make_note_excerpt(content))

    def _build_preview_meta(self, row: ReportRow | None) -> str:
        if row is None:
            return "Пока нет review-истории. Можно начать с ручного review и заметка появится в приоритетной очереди."

        if row.urgency >= 0.92:
            recommendation = "Нужна немедленная проработка — это хорошая кандидатура для следующего повторения."
        elif row.urgency >= 0.80:
            recommendation = "Лучше повторить сегодня, пока не накопился хвост."
        elif row.urgency >= 0.65:
            recommendation = "Имеет смысл взять в ближайшую учебную сессию."
        elif row.urgency >= 0.50:
            recommendation = "Можно повторить после красных и оранжевых заметок."
        else:
            recommendation = "Состояние стабильное, заметка не требует срочного внимания."

        return (
            f"Срочность: {row.urgency:.2f} · band: {row.band} · повторов: {row.review_count} · "
            f"последняя сложность: {row.last_difficulty} · последнее повторение: {format_runtime(row.last_ts, fallback='—')}\n"
            f"{recommendation}"
        )

    def _write_preview_text(self, text: str) -> None:
        for widget in self.preview_text_widgets:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")

    def _make_note_excerpt(self, content: str, limit: int = 1600) -> str:
        lines = content.splitlines()
        if lines and lines[0].strip() == "---":
            end_idx = None
            for idx in range(1, len(lines)):
                if lines[idx].strip() == "---":
                    end_idx = idx
                    break
            if end_idx is not None:
                lines = lines[end_idx + 1 :]

        cleaned = [line.rstrip() for line in lines]
        text = "\n".join(cleaned).strip()
        if not text:
            return "Заметка пустая."
        if len(text) > limit:
            return text[:limit].rstrip() + "\n…"
        return text

    def _copy_selected_note_path(self) -> None:
        if not self.preview_note_path:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.preview_note_path)
        self.status_var.set("Путь к заметке скопирован.")

    def _clear_report_filters(self) -> None:
        self.report_search_var.set("")
        self.report_band_var.set("Все уровни")
        self.report_limit_var.set(DEFAULT_REPORT_LIMIT)
        self.sort_column = "urgency"
        self.sort_descending = True
        self._update_report_headings()
        self._apply_report_filters()

    def run_scan(self) -> None:
        def worker(vault: Path, run_at: datetime, emit: Callable[[str], None]) -> dict[str, object]:
            store = SRSStore(vault)
            count = scan_and_index(vault, store, run_at, emit=emit)
            refreshed_store = SRSStore(vault)
            rows = self._build_full_report(vault, refreshed_store, run_at)
            synced_rows, queue_summary = sync_obsidian_bridge(vault, refreshed_store, run_at, rows=rows, emit=emit)
            return {
                "count": count,
                "rows": synced_rows,
                "queue_summary": {
                    "path": queue_summary.path,
                    "row_count": queue_summary.row_count,
                    "untracked_count": queue_summary.untracked_count,
                },
            }

        self._run_background_action("scan", "Сканирование маркеров", worker)

    def run_sync(self) -> None:
        def worker(vault: Path, run_at: datetime, emit: Callable[[str], None]) -> dict[str, object]:
            store = SRSStore(vault)
            count = sync_tags(vault, store, run_at, emit=emit)
            refreshed_store = SRSStore(vault)
            rows = self._build_full_report(vault, refreshed_store, run_at)
            synced_rows, queue_summary = sync_obsidian_bridge(vault, refreshed_store, run_at, rows=rows, emit=emit)
            return {
                "count": count,
                "rows": synced_rows,
                "queue_summary": {
                    "path": queue_summary.path,
                    "row_count": queue_summary.row_count,
                    "untracked_count": queue_summary.untracked_count,
                },
            }

        self._run_background_action("sync", "Синхронизация тегов", worker)

    def run_report(self) -> None:
        def worker(vault: Path, run_at: datetime, emit: Callable[[str], None]) -> dict[str, object]:
            store = SRSStore(vault)
            rows = self._build_full_report(vault, store, run_at)
            emit(f"[report] rebuilt {len(rows)} rows")
            synced_rows, queue_summary = sync_obsidian_bridge(vault, store, run_at, rows=rows, emit=emit)
            return {
                "rows": synced_rows,
                "queue_summary": {
                    "path": queue_summary.path,
                    "row_count": queue_summary.row_count,
                    "untracked_count": queue_summary.untracked_count,
                },
            }

        self._run_background_action("report", "Построение отчёта", worker)

    def run_manual_review(self) -> None:
        note_path = self.review_note_var.get().strip()
        if not note_path:
            messagebox.showerror("Ручное ревью", "Выберите заметку или введите относительный путь к ней.")
            return

        difficulty = self._safe_int(self.review_difficulty_var.get(), 5, minimum=1, maximum=10)

        def worker(vault: Path, run_at: datetime, emit: Callable[[str], None]) -> dict[str, object]:
            store = SRSStore(vault)
            add_manual_review(vault, store, note_path, difficulty, run_at, emit=emit)
            refreshed_store = SRSStore(vault)
            rows = self._build_full_report(vault, refreshed_store, run_at)
            synced_rows, queue_summary = sync_obsidian_bridge(vault, refreshed_store, run_at, rows=rows, emit=emit)
            return {
                "rows": synced_rows,
                "queue_summary": {
                    "path": queue_summary.path,
                    "row_count": queue_summary.row_count,
                    "untracked_count": queue_summary.untracked_count,
                },
            }

        self._run_background_action("review", "Добавление ручного ревью", worker)


def main() -> None:
    root = tk.Tk()
    app = SRSDesktopApp(root)
    app._update_report_headings()
    app._append_log("Интерфейс готов.")
    root.mainloop()


if __name__ == "__main__":
    main()
