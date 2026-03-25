#!/usr/bin/env python3
import streamlit as st
import os
import re
import urllib.parse
import webbrowser
import pandas as pd

# ---------- Configuration: put your files here ----------
TELUGU_CSV = "telugu_songs1_dataset.csv"
HINDI_CSV = "hindi_songs_dataset.csv"

# ---------- Utilities ----------
def read_csv_try(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1")
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV {path}: {e}")

def detect_mood_col(df):
    if df is None or df.empty:
        return None
    candidates = ['mood','moods','feeling','feels','song_mood','emotion','sentiment']
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_map:
            return cols_map[cand]
    for low, orig in cols_map.items():
        if 'mood' in low or 'feel' in low or 'sent' in low or 'emotion' in low:
            return orig
    return None

def detect_link_col(df):
    if df is None or df.empty:
        return None
    candidates = ['youtube_link','youtube','link','url','video_url']
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_map:
            return cols_map[cand]
    for low, orig in cols_map.items():
        if 'youtube' in low or 'link' in low or 'url' in low:
            return orig
    return None

def normalize_mood_series(df, mood_col):
    if mood_col is None:
        return pd.Series([''] * len(df))
    return df[mood_col].astype(str).str.strip().str.lower().fillna('')

def is_youtube_id(s):
    return bool(re.fullmatch(r'[A-Za-z0-9_-]{11}', s))

def fix_telugu_link(val):
    if pd.isna(val):
        return ''
    s = str(val).strip()
    if not s:
        return ''
    if re.match(r'^https?://', s, flags=re.I):
        return s
    if is_youtube_id(s):
        return f'https://youtu.be/{s}'
    if s.startswith('www.'):
        return 'https://' + s
    if 'youtube' in s or 'youtu.be' in s:
        return 'https://' + s
    if '.' in s and ' ' not in s:
        return 'https://' + s
    return ''

def make_youtube_search_link(title, artist=None, movie=None):
    parts = []
    if title:
        parts.append(str(title))
    if artist:
        parts.append(str(artist))
    if movie:
        parts.append(str(movie))
    q = " ".join(parts).strip()
    if not q:
        return ''
    return 'https://www.youtube.com/results?search_query=' + urllib.parse.quote_plus(q)

def first_col_like(df, possibilities):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lowmap = {c.lower(): c for c in cols}
    for p in possibilities:
        if p.lower() in lowmap:
            return lowmap[p.lower()]
    for c in cols:
        if 'title' in c.lower() or 'song' in c.lower() or 'name' in c.lower():
            return c
    return cols[0] if cols else None

# ---------- Prepare dataset helper ----------
def prepare_dataset(df, dataset_name):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Title","Artist","Movie","Mood","mood_norm","Link","Dataset"])
    df = df.copy()
    mood_col = detect_mood_col(df)
    if mood_col is None:
        raise RuntimeError(f"Could not detect mood column in dataset '{dataset_name}'")
    df["mood_norm"] = normalize_mood_series(df, mood_col)
    title_col = first_col_like(df, ['song_name','title','song','track','name'])
    # artist detection
    artist_col = None
    for c in df.columns:
        if 'artist' in c.lower() or 'singer' in c.lower():
            artist_col = c
            break
    movie_col = None
    for c in df.columns:
        if any(x in c.lower() for x in ("movie","film","album")):
            movie_col = c
            break
    link_col = detect_link_col(df)
    if link_col is None:
        df["auto_link"] = df.apply(lambda r: make_youtube_search_link(r.get(title_col,""), r.get(artist_col,"") if artist_col else "", r.get(movie_col,"") if movie_col else ""), axis=1)
        link_col = "auto_link"
    else:
        df[link_col] = df[link_col].astype(str).apply(fix_telugu_link)
    idx = df.index
    title_s = df[title_col].astype(str) if title_col in df.columns else pd.Series("", index=idx)
    artist_s = df[artist_col].astype(str) if artist_col and artist_col in df.columns else pd.Series("", index=idx)
    movie_s = df[movie_col].astype(str) if movie_col and movie_col in df.columns else pd.Series("", index=idx)
    mood_s = df[mood_col].astype(str) if mood_col in df.columns else pd.Series("", index=idx)
    mood_norm_s = df["mood_norm"]
    link_s = df[link_col].astype(str) if link_col in df.columns else pd.Series("", index=idx)
    clean = pd.DataFrame({
        "Title": title_s.values,
        "Artist": artist_s.values,
        "Movie": movie_s.values,
        "Mood": mood_s.values,
        "mood_norm": mood_norm_s.values,
        "Link": link_s.values,
        "Dataset": [dataset_name] * len(df)
    })
    return clean

# ---------- Load CSVs at startup (no UI load buttons) ----------
tel_df = pd.DataFrame()
hin_df = pd.DataFrame()
tel_loaded = False
hin_loaded = False

if os.path.exists(TELUGU_CSV):
    try:
        tel_df = read_csv_try(TELUGU_CSV)
        tel_loaded = True
    except Exception as e:
        print(f"Warning: could not load Telugu CSV: {e}")

if os.path.exists(HINDI_CSV):
    try:
        hin_df = read_csv_try(HINDI_CSV)
        hin_loaded = True
    except Exception as e:
        print(f"Warning: could not load Hindi CSV: {e}")

telugu_clean = prepare_dataset(tel_df, "Telugu") if tel_loaded else pd.DataFrame()
hindi_clean = prepare_dataset(hin_df, "Hindi") if hin_loaded else pd.DataFrame()

combined = pd.concat([telugu_clean, hindi_clean], ignore_index=True)
combined = combined.drop_duplicates(subset=["Title"], keep="first").reset_index(drop=True)
all_moods = sorted([m for m in combined["mood_norm"].unique() if str(m).strip()])

# ---------- GUI ----------
root = tk.Tk()
root.title("Mood-Based Song Recommender")
root.geometry("1000x650")
style = ttk.Style(root)
try:
    style.theme_use('clam')
except Exception:
    pass

container = ttk.Frame(root, padding=12)
container.pack(fill="both", expand=True)
container.rowconfigure(0, weight=1)
container.columnconfigure(0, weight=1)

pages = {}
def add_page(name, frame):
    pages[name] = frame
    frame.grid(row=0, column=0, sticky="nsew")
def show(name):
    pages[name].tkraise()

# ---------- Page 1: Input (NO load buttons) ----------
page1 = ttk.Frame(container, padding=20)
add_page("input", page1)
center = ttk.Frame(page1)
center.place(relx=0.5, rely=0.5, anchor="center")

ttk.Label(center, text="🎧 Mood-Based Song Recommender", font=("Arial", 20, "bold")).pack(pady=(0,8))
ttk.Label(center, text="Select your mood (detected from datasets) or type one.", font=("Arial", 11)).pack(pady=(0,8))

mood_var = tk.StringVar()
if all_moods:
    mood_widget = ttk.Combobox(center, textvariable=mood_var, values=all_moods, width=50, state="normal")
else:
    mood_widget = ttk.Entry(center, textvariable=mood_var, width=52)
mood_widget.pack(pady=8)

def get_recommendations():
    sel = (mood_var.get() or "").strip().lower()
    if not sel:
        messagebox.showerror("No mood", "Please select or type a mood.")
        return
    tel_matches = combined[(combined["mood_norm"] == sel) & (combined["Dataset"] == "Telugu")]
    hin_matches = combined[(combined["mood_norm"] == sel) & (combined["Dataset"] == "Hindi")]
    final = pd.concat([tel_matches, hin_matches], ignore_index=True)
    final = final.drop_duplicates(subset=["Title"], keep="first").reset_index(drop=True)
    if final.empty:
        messagebox.showinfo("No results", f"No songs found for mood '{sel}'.")
        return
    fill_table(final)
    show("output")

btn_frame = ttk.Frame(center)
btn_frame.pack(pady=10)
ttk.Button(btn_frame, text="Get Recommendations", command=get_recommendations).pack(side="left", padx=6)

lbl_info = ttk.Label(center, text=("Detected moods: " + ", ".join(all_moods[:20])) if all_moods else "No moods detected yet.", foreground="gray", wraplength=700, justify="center")
lbl_info.pack(pady=(8,0))

# ---------- Page 2: Output ----------
page2 = ttk.Frame(container, padding=12)
add_page("output", page2)
ttk.Label(page2, text="🎵 Recommended Songs", font=("Arial", 18, "bold")).pack(pady=(6,8))

table_frame = ttk.Frame(page2)
table_frame.pack(fill="both", expand=True, padx=8, pady=6)
cols = ("#","Title","Artist","Movie","Mood","Dataset","Link")
tree = ttk.Treeview(table_frame, columns=cols, show="headings")
for c in cols:
    tree.heading(c, text=c)
tree.column("#", width=40, anchor="center")
tree.column("Title", width=300, anchor="w")
tree.column("Artist", width=180, anchor="w")
tree.column("Movie", width=150, anchor="w")
tree.column("Mood", width=90, anchor="center")
tree.column("Dataset", width=80, anchor="center")
tree.column("Link", width=260, anchor="w")

vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
tree.grid(row=0, column=0, sticky="nsew")
vsb.grid(row=0, column=1, sticky="ns")
hsb.grid(row=1, column=0, sticky="ew")
table_frame.rowconfigure(0, weight=1)
table_frame.columnconfigure(0, weight=1)

def fill_table(df):
    for r in tree.get_children():
        tree.delete(r)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        link_val = row.get("Link","") or ""
        if row.get("Dataset","") == "Telugu" and not link_val:
            link_val = make_youtube_search_link(row.get("Title",""), row.get("Artist",""), row.get("Movie",""))
        tree.insert("", "end", values=(i, row.get("Title",""), row.get("Artist",""), row.get("Movie",""), row.get("Mood",""), row.get("Dataset",""), link_val))

def open_selected_link(event=None):
    sel = tree.selection()
    if not sel:
        messagebox.showinfo("Select", "Please select a row first.")
        return
    vals = tree.item(sel[0]).get("values", [])
    if not vals:
        return
    url = vals[-1]
    if not url:
        messagebox.showinfo("No link", "No link available for this selection.")
        return
    try:
        webbrowser.open_new_tab(url)
    except Exception as e:
        messagebox.showerror("Open error", f"Could not open URL: {e}")

tree.bind("<Double-1>", open_selected_link)

btn_frame2 = ttk.Frame(page2)
btn_frame2.pack(pady=8)
ttk.Button(btn_frame2, text="⬅ Back", command=lambda: show("input")).pack(side="left", padx=6)
ttk.Button(btn_frame2, text="Open Selected Link", command=open_selected_link).pack(side="left", padx=6)

# Start app on input page
show("input")

root.mainloop()

