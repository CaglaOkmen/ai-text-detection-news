import pandas as pd
import string
import re
import numpy as np
import os
import math
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import stanza

OUTPUT_DIR  = "makale_gorseller"
CACHE_FILE  = "oznitelik_onbellek.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. VERİ YÜKLEME ────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("haber_veri_seti.csv", encoding="utf-8-sig")
    print("=" * 60)
    print(f"[BAŞARILI] Veri seti yüklendi. Toplam satır: {df.shape[0]}")
    print(df['etiket'].value_counts().to_string())
    print("=" * 60 + "\n")
except Exception as e:
    print(f"[HATA] {e}"); exit()

# ── 2. BİRLEŞTİRİLMİŞ VE GENİŞLETİLMİŞ ÖZNİTELİK KÜMESİ (11 Boyutlu) ───────────
FEATURE_NAMES = [
    'Toplam Kelime',
    'Ort. Kelime Uzunluğu',
    'Ort. Cümle Uzunluğu',
    'Kelime Zenginliği (TTR)',
    'Toplam Bağlaç Yoğunluğu',          # CCONJ + SCONJ + ADP
    'Yapısal Kelime Yoğunluğu',         # ADV + ADJ + DET
    'İsim Yoğunluğu',                   # PROPN + NOUN
    'Sayısal Veri Yoğunluğu (NUM)',     
    'Noktalama Yoğunluğu',
    'Metin Entropisi (Shannon)',        # Yeni Öznitelik 1
    'Cümle Uzunluğu Standart Sapması'   # Yeni Öznitelik 2
]

# ── 3. YARDIMCI DİLBİLİMSEL FONKSİYONLAR ─────────────────────────────────────
def metin_entropisi(words):
    """Metindeki kelime dağılımının düzensizliğini (Shannon Entropisi) ölçer."""
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy

# ── 4. ÖZNİTELİK FONKSİYONU ───────────────────────────────────────────────────
def ozellik_muhendisligi(text):
    """
    Metinden makro kategoriler altında birleştirilmiş ve adli dilbilimsel 
    entropi/sapma metrikleriyle zenginleştirilmiş 11 adet öznitelik çıkarır.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return [0.0] * len(FEATURE_NAMES)

    text_clean = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text_clean.split()
    toplam_kelime   = max(len(words), 1)
    toplam_karakter = len(text)
    ttr             = len(set(words)) / toplam_kelime
    noktalama_orani = sum(1 for c in text if c in string.punctuation) / toplam_kelime

    # Yeni Metrik 1: Shannon Kelime Entropisi
    entropy_degeri = metin_entropisi(words)

    baglac_sayisi   = 0  
    yapisal_sayisi  = 0  
    isim_sayisi     = 0  
    num_sayisi      = 0  
    toplam_cumle    = 1
    cumle_uzunluklari = []

    try:
        doc = nlp(text)
        toplam_cumle = max(len(doc.sentences), 1)

        for sentence in doc.sentences:
            sentence_words = sentence.words
            
            # Yeni Metrik 2 İçin Hazırlık: Her cümlenin kelime sayısını hesaplama
            clean_sentence_words = [w for w in sentence_words if isinstance(w.id, int) and w.upos != 'PUNCT']
            cumle_uzunluklari.append(max(len(clean_sentence_words), 1))

            for idx, word in enumerate(sentence_words):
                if not isinstance(word.id, int):
                    continue
                
                upos = word.upos
                feats = word.feats if word.feats else ""
                word_text = word.text.lower()
                
                # 1. KURAL: Gelişmiş Sayısal Veri Kontrolü (NUM)
                if upos == 'NUM' or re.match(r'^\d+([.,]\d+)?$', word_text) or (upos == 'X' and any(c.isdigit() for c in word_text)):
                    num_sayisi += 1
                    continue

                # 2. KURAL: "bir" Kelimesinin Bağlamsal Belirteç (DET) Kontrolü
                is_det_one = False
                if word_text == 'bir' and idx + 1 < len(sentence_words):
                    if sentence_words[idx + 1].upos in ['NOUN', 'ADJ']:
                        is_det_one = True

                # KATEGORİ A: Toplam Bağlaç Yapısı (CCONJ + SCONJ + ADP)
                if upos in ['CCONJ', 'ADP', 'SCONJ'] or 'VerbForm=Conv' in feats:
                    baglac_sayisi += 1

                # KATEGORİ B: Yapısal Kelimeler (ADV + ADJ + DET)
                elif upos in ['ADJ', 'DET', 'ADV'] or 'Case=Ins' in feats or is_det_one:
                    yapisal_sayisi += 1

                # KATEGORİ C: İsim Soylu İçerik Kelimeleri (NOUN + PROPN)
                elif upos in ['NOUN', 'PROPN']:
                    isim_sayisi += 1

    except Exception:
        sentences_fallback = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        toplam_cumle = max(len(sentences_fallback), 1)
        cumle_uzunluklari = [len(s.split()) for s in sentences_fallback] if sentences_fallback else [toplam_kelime]

    avg_word_len     = toplam_karakter / toplam_kelime
    avg_sentence_len = toplam_kelime   / toplam_cumle
    
    baglac_orani   = baglac_sayisi   / toplam_kelime
    yapisal_orani  = yapisal_sayisi  / toplam_kelime
    isim_orani     = isim_sayisi     / toplam_kelime
    num_orani      = num_sayisi      / toplam_kelime

    # Yeni Metrik 2: Cümle Uzunlukları Standart Sapması Hesaplama
    std_sentence_len = float(np.std(cumle_uzunluklari)) if len(cumle_uzunluklari) > 0 else 0.0

    return [
        toplam_kelime, avg_word_len, avg_sentence_len, ttr,
        baglac_orani, yapisal_orani, isim_orani, num_orani, 
        noktalama_orani, entropy_degeri, std_sentence_len
    ]

# ── 5. STANZA + ÖNBELLEK KONTROLÜ ─────────────────────────────────────────────
if os.path.exists(CACHE_FILE):
    cache_df = pd.read_csv(CACHE_FILE)
    if len(cache_df) == len(df) and list(cache_df.columns) == FEATURE_NAMES:
        print(f"[BİLGİ] Önbellek uyumlu bulundu → '{CACHE_FILE}' yükleniyor...")
        X = cache_df
        y = df['etiket']
    else:
        print("[BİLGİ] Önbellek eski veya uyumsuz. Yeniden hesaplanıyor...")
        try: os.remove(CACHE_FILE)
        except: pass
        os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(CACHE_FILE):
    print("[İşlem] Stanza Türkçe modeli yükleniyor...")
    stanza.download('tr', processors='tokenize,mwt,pos,lemma,depparse')
    nlp = stanza.Pipeline('tr', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True, verbose=False)
    print("[İşlem] 11 boyutlu genişletilmiş öznitelikler çıkarılıyor (Lütfen bekleyin)...")
    X = pd.DataFrame(df['metin'].apply(ozellik_muhendisligi).tolist(), columns=FEATURE_NAMES)
    X.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
    print(f"[BAŞARILI] Öznitelik seti kaydedildi → '{CACHE_FILE}'\n")
    y = df['etiket']

# ── 6. EĞİTİM / TEST BÖLMESİ ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 7. MODELLER VE EĞİTİM ──────────────────────────────────────────────────────
modeller = {
    "SVM":   SVC(kernel='linear', probability=True, random_state=20),
    "XGB":   XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.01, random_state=20, eval_metric='logloss'), 
    "LR":    LogisticRegression(max_iter=1000, random_state=20),
}
model_etiketleri = {
    "SVM":   "Destek Vektör Makineleri (SVM)",
    "XGB":   "eXtreme Gradient Boosting (XGBoost)",
    "LR":    "Lojistik Regresyon (LR)",
}

sonuclar = {}; cm_dict = {}; cv_dict = {}; roc_dict = {}
renkler  = ['#2196F3', '#4CAF50', '#FF5722']

print("[İşlem] Modeller eğitiliyor...\n")
for kisa, model in modeller.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    rapor = classification_report(y_test, y_pred, output_dict=True)
    sonuclar[kisa] = {
        "Doğruluk (Accuracy)":  round(rapor["accuracy"], 4),
        "Kesinlik (Precision)": round(rapor["macro avg"]["precision"], 4),
        "Duyarlılık (Recall)":  round(rapor["macro avg"]["recall"], 4),
        "F1-Skor":              round(rapor["macro avg"]["f1-score"], 4),
    }
    cm_dict[kisa]  = confusion_matrix(y_test, y_pred)
    cv_dict[kisa]  = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='f1_macro')
    fpr, tpr, _    = roc_curve(y_test, y_prob)
    roc_dict[kisa] = (fpr, tpr, auc(fpr, tpr))

    print(f"{'='*55}\n  {model_etiketleri[kisa]}\n{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["İnsan (0)", "Yapay Zeka (1)"]))
    print(f"  5-katlı CV F1: {cv_dict[kisa].mean():.4f} ± {cv_dict[kisa].std():.4f}\n")

# ── 8. TXT RAPORU ─────────────────────────────────────────────────────────────
rows = []
for k in modeller:
    rows.append({"Model": model_etiketleri[k], **sonuclar[k],
                 "CV_F1_Ort": round(cv_dict[k].mean(), 4),
                 "CV_F1_Std": round(cv_dict[k].std(), 4)})

with open(os.path.join(OUTPUT_DIR, "detayli_rapor.txt"), "w", encoding="utf-8") as f:
    f.write("YAPAY ZEKA METİN TESPİTİ PERFORMANS RAPORU (GENİŞLETİLMİŞ SET)\n" + "="*75 + "\n\n")
    f.write(pd.DataFrame(rows).to_string(index=False) + "\n\n")
    f.write("5-Katlı CV Fold Detayları:\n")
    for k in modeller:
        folds = ", ".join([f"F{i+1}:{s:.4f}" for i, s in enumerate(cv_dict[k])])
        f.write(f"  {model_etiketleri[k]}: {folds}\n")
    f.write("\n")
    
    f.write("------------------------------------------------------------\n")
    f.write("DİNAMİK MODEL DOĞRULAMA - İLK 3 METNİN HESAPLANAN SONUÇLARI\n")
    f.write("------------------------------------------------------------\n")
    for idx in range(min(3, len(df))):
        sınıf_adı = "Yapay Zeka (1)" if df['etiket'].iloc[idx] == 1 else "İnsan (0)"
        metin_ozet = df['metin'].iloc[idx][:120] + "..."
        f.write(f"Veri İndeksi: {idx} | Sınıfı: {sınıf_adı}\n")
        f.write(f"Metin Özeti: \"{metin_ozet}\"\n\n")
        f.write("Hesaplanan Öznitelik Değerleri:\n")
        for feat in FEATURE_NAMES:
            f.write(f"  - {feat.ljust(42)}: {X[feat].iloc[idx]:.4f}\n")
        f.write("-" * 45 + "\n")

print(f"[DOSYA KAYDEDİLDİ] {OUTPUT_DIR}/detayli_rapor.txt\n")

# ── 9. GRAFİKLER ──────────────────────────────────────────────────────────────

# Şekil 1 — Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Şekil 1. Sınıflandırıcılara Ait Karmaşıklık Matrisleri", fontsize=13, fontweight='bold', y=1.02)
for ax, (kisa, cm) in zip(axes, cm_dict.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['İnsan (0)', 'YZ (1)'], yticklabels=['İnsan (0)', 'YZ (1)'], annot_kws={"size": 13})
    ax.set_title(model_etiketleri[kisa], fontsize=11, pad=8)
    ax.set_ylabel('Gerçek Sınıf', fontsize=10); ax.set_xlabel('Tahmin Edilen', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sekil1_confusion_matrices.png"), dpi=300, bbox_inches='tight')
plt.close()

# Şekil 2 — Performans Çubuğu
metrikler = ["Doğruluk (Accuracy)", "Kesinlik (Precision)", "Duyarlılık (Recall)", "F1-Skor"]
x = np.arange(len(metrikler)); width = 0.22
fig, ax = plt.subplots(figsize=(11, 5))
for i, (kisa, renk) in enumerate(zip(modeller.keys(), renkler)):
    degerler = [sonuclar[kisa][m] for m in metrikler]
    bars = ax.bar(x + i*width, degerler, width, label=model_etiketleri[kisa], color=renk, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, degerler):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)
ax.set_title("Şekil 2. Sınıflandırıcı Performans Karşılaştırması", fontsize=13, fontweight='bold')
ax.set_xticks(x + width); ax.set_xticklabels(metrikler, fontsize=10)
ax.set_ylim(0, 1.12); ax.legend(loc='lower right', fontsize=9)
ax.yaxis.grid(True, linestyle='--', alpha=0.6); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sekil2_performans_karsilastirma.png"), dpi=300, bbox_inches='tight')
plt.close()

# Şekil 3 — ROC
fig, ax = plt.subplots(figsize=(7, 5))
for (kisa, (fpr, tpr, roc_auc)), renk in zip(roc_dict.items(), renkler):
    ax.plot(fpr, tpr, color=renk, lw=2, label=f"{model_etiketleri[kisa]} (AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1,label='Rastgele Sınıflandırıcı')
ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
ax.set_title("Şekil 3. ROC Eğrileri ve AUC Değerleri", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9); ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sekil3_roc_egrileri.png"), dpi=300, bbox_inches='tight')
plt.close()

# Şekil 4 — Fold Bazlı CV
fig, ax = plt.subplots(figsize=(8, 5))
katmanlar = ["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5"]
for (kisa, renk) in zip(modeller.keys(), renkler):
    ax.plot(katmanlar, cv_dict[kisa], marker='o', markersize=6, linewidth=2, color=renk, label=f"{model_etiketleri[kisa]} (Ort: {cv_dict[kisa].mean():.3f})")
ax.set_title("Şekil 4. Katman Bazlı Çapraz Doğrulama (5-Fold) F1-Skoru", fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05]); ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.xaxis.grid(True, linestyle=':', alpha=0.4)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sekil4_cv_fold_dagilimi.png"), dpi=300, bbox_inches='tight')
plt.close()

# Şekil 5'ler — Her model için Öznitelik Önemi (güncellendi)
print("[İşlem] Her model için öznitelik önem grafikleri oluşturuluyor...")
for kisa, model in modeller.items():
    # Model bazlı önem puanı hesaplama
    if kisa == "XGB":
        importances = model.feature_importances_
    elif kisa in ["LR", "SVM"]:
        # Lineer modeller için katsayıların mutlak değeri (binary durum)
        importances = np.abs(model.coef_[0])
    
    sorted_idx = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh([FEATURE_NAMES[i] for i in sorted_idx], importances[sorted_idx], color='#42A5F5', edgecolor='white')
    
    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    
    ax.set_title(f"Şekil 5. {model_etiketleri[kisa]} — Öznitelik Önem Sıralaması", fontsize=13, fontweight='bold')
    ax.xaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
    plt.tight_layout()
    
    # Model bazlı dosya adı
    filename = f"sekil5_{kisa.lower()}_ozellik_onemi.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Kaydedildi: {OUTPUT_DIR}/{filename}")

# Şekil 6 — Öznitelik Dağılımı
ncols = 4; nrows = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
axes_flat = axes.flatten()
sinif_renk   = {0: '#1E88E5', 1: '#E53935'}
sinif_etiket = {0: 'İnsan',   1: 'Yapay Zeka'}
for i, feat in enumerate(FEATURE_NAMES):
    ax = axes_flat[i]
    for sinif, renk in sinif_renk.items():
        ax.hist(X[feat][y == sinif], bins=20, alpha=0.65, color=renk, label=sinif_etiket[sinif], edgecolor='white')
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
for j in range(len(FEATURE_NAMES), len(axes_flat)):
    axes_flat[j].set_visible(False)
fig.suptitle("Şekil 6. Sınıf Bazlı Öznitelik Dağılımları", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sekil6_ozellik_dagilimi.png"), dpi=300, bbox_inches='tight')
plt.close()

# ── 10. ŞEKİL 7 — EN ÖNEMLİ İLK 6 ÖZNİTELİK DAĞILIMI (HER MODEL İÇİN GÜNCELLENDİ) ───────────
print("\n[İşlem] Her model için en önemli ilk 6 öznitelik dağılım grafikleri oluşturuluyor...")

for kisa, model in modeller.items():
    if kisa == "XGB":
        importances = model.feature_importances_
    elif kisa in ["LR", "SVM"]:
        importances = np.abs(model.coef_[0])
    
    en_onemli_6_idx = np.argsort(importances)[::-1][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, idx in enumerate(en_onemli_6_idx):
        ax = axes_flat[i]
        feat = FEATURE_NAMES[idx]
        for sinif, renk in sinif_renk.items():
            ax.hist(X[feat][y == sinif], bins=25, alpha=0.7, color=renk, label=sinif_etiket[sinif], edgecolor='white')
        ax.set_title(f"Önem Sıralaması #{i+1}: {feat}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Öznitelik Değeri", fontsize=9)
        ax.set_ylabel("Döküman Frekansı", fontsize=9)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle(f"Şekil 7. {model_etiketleri[kisa]} Karar Mekanizmasını Belirleyen En Kritik İlk 6 Özniteliğin Sınıf Bazlı Dağılımları", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Model bazlı dosya adı
    filename = f"sekil7_{kisa.lower()}_6_dagilim.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Kaydedildi: {OUTPUT_DIR}/{filename}")

print(f"\nTüm model bazlı ve güncellenmiş makro öznitelik çıktıları '{OUTPUT_DIR}/' klasörüne aktarıldı.")