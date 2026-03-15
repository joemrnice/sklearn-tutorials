# Scikit-learn Mastery Course (DOCX Build)

## Build a DOCX (Academic style)

### 1) Install Pandoc
- macOS: `brew install pandoc`
- Windows: install from Pandoc releases
- Linux: use your package manager (may be slightly older) or install from releases

### 2) (Recommended) Create an Academic reference.docx template
```bash
pandoc --print-default-data-file reference.docx > course/pandoc/reference.docx
```

Open `course/pandoc/reference.docx` in Word and adjust styles:
- Normal: Cambria 11pt, 1.15–1.5 spacing
- Headings: consistent spacing and hierarchy
- Code: Consolas 9–10pt

### 3) Build
```bash
bash tools/make_docx.sh
```

Output:
- `dist/Scikit-learn_Mastery.docx`
- `dist/Scikit-learn_Mastery.build.md` (concatenated source used for the build)