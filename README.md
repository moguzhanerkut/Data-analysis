# Bank Marketing - Data Science Project

## 📊 Proje Açıklaması

Bu proje, Bank Marketing veri seti üzerinde kapsamlı bir veri madenciliği analizi gerçekleştirir. Müşterilerin vadeli mevduat hesabı açıp açmayacağını tahmin etmek için çeşitli makine öğrenmesi algoritmaları uygulanmıştır.

## 📁 Dosya Yapısı

```
320/
├── bank-additional-full.csv          # Veri seti
├── preprocessing.py                   # Veri ön işleme modülü
├── visualization.py                   # Görselleştirme modülü
├── classification_models.py           # Sınıflandırma modelleri
├── regression_models.py               # Regresyon modelleri
├── clustering_models.py               # Kümeleme modelleri
├── results_analysis.py                # Sonuç analizi modülü
├── bank_marketing_analysis.ipynb      # Ana Jupyter Notebook
├── README.md                          # Bu dosya
├── results_report.txt                 # Otomatik oluşturulan rapor
└── visualizations/                    # Tüm görselleştirmeler
    ├── 01_age_distribution.png
    ├── 02_target_distribution.png
    ├── 03_correlation_heatmap.png
    └── ... (20+ görselleştirme)
```

## 🚀 Hızlı Başlangıç

### Gereksinimler

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

### Jupyter Notebook ile Çalıştırma (Önerilen)

```bash
jupyter notebook bank_marketing_analysis.ipynb
```

Notebook'u açtıktan sonra "Cell" > "Run All" seçeneğini tıklayın.

### Python Modüllerini Tek Tek Çalıştırma

```bash
# 1. Veri ön işleme
python preprocessing.py

# 2. Görselleştirmeler
python visualization.py

# 3. Sınıflandırma modelleri
python classification_models.py

# 4. Regresyon modelleri
python regression_models.py

# 5. Kümeleme modelleri
python clustering_models.py
```

## 📈 Proje Kapsamı

### 1. Veri Ön İşleme
- ✅ Eksik değerlerin işlenmesi ("unknown" → mode)
- ✅ Outlier temizleme (IQR metodu)
- ✅ Kategorik encoding (Label Encoding)
- ✅ Normalizasyon (StandardScaler)
- ✅ PCA (gerekirse)
- ✅ Train-test split (%80-20)

### 2. Görselleştirme (6+ grafik)
- ✅ Yaş dağılımı
- ✅ Hedef değişken dağılımı
- ✅ Korelasyon ısı haritası
- ✅ Meslek vs hedef değişken
- ✅ Kampanya analizi
- ✅ Eğitim ve medeni durum analizi

### 3. Sınıflandırma Modelleri
- ✅ **Decision Tree** (max_depth=10)
- ✅ **Random Forest** (n_estimators=100)
- ✅ **Logistic Regression** (max_iter=1000)

**Metrikler:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

### 4. Regresyon Modelleri
- ✅ **Linear Regression**
- ✅ **Ridge Regression** (alpha=1.0)

**Hedef:** `duration` (görüşme süresi)  
**Metrikler:** MSE, RMSE, MAE, R² Score

### 5. Kümeleme Analizi
- ✅ **K-Means** (optimal k bulma)
- ✅ **Hierarchical Clustering** (Ward linkage)

**Metrikler:** Silhouette Score, Davies-Bouldin Index, Inertia

### 6. Sonuç Analizi
- ✅ Tüm modellerin karşılaştırılması
- ✅ Kapsamlı rapor oluşturma
- ✅ Görselleştirmeler
- ✅ En iyi modellerin belirlenmesi

## 📊 Beklenen Sonuçlar

### Sınıflandırma
- **En İyi Model:** Random Forest
- **Tipik Accuracy:** 0.85-0.92
- **Tipik F1-Score:** 0.80-0.88

### Regresyon
- **Tipik R² Score:** 0.30-0.45
- **Tipik RMSE:** 180-220 saniye

### Kümeleme
- **Optimal Küme Sayısı:** 3-5
- **Tipik Silhouette Score:** 0.25-0.35

## 💡 Önemli Bulgular

1. **Görüşme Süresi:** Başarının en önemli göstergesi
2. **Sınıf Dengesizliği:** Çoğunlukla "no" yanıtı var
3. **Ekonomik Göstergeler:** Kampanya başarısını etkiliyor
4. **Müşteri Segmentleri:** 3-5 farklı segment tespit edildi

## 🎯 Öneriler

### Teknik
- SMOTE ile sınıf dengesizliğini giderin
- GridSearchCV ile hyperparameter tuning yapın
- Feature engineering ile yeni özellikler türetin

### İş
- Uzun görüşmelere odaklanın
- Belirlenen segmentleri hedefleyin
- Ekonomik göstergeleri kampanya zamanlaması için kullanın
- Random Forest tahminlerini karar destek sistemi olarak kullanın

## 📚 Modül Açıklamaları

### `preprocessing.py`
Veri yükleme, temizleme, encoding, normalizasyon ve PCA işlemlerini gerçekleştirir.

**Kullanım:**
```python
from preprocessing import load_and_preprocess_data
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
```

### `visualization.py`
Keşifsel veri analizi için görselleştirmeler oluşturur.

**Kullanım:**
```python
from visualization import DataVisualizer
visualizer = DataVisualizer(df)
visualizer.generate_all_basic_visualizations()
```

### `classification_models.py`
Sınıflandırma modellerini eğitir ve değerlendirir.

**Kullanım:**
```python
from classification_models import ClassificationModels
classifier = ClassificationModels(X_train, X_test, y_train, y_test)
classifier.train_and_evaluate_all()
```

### `regression_models.py`
Regresyon modellerini eğitir ve değerlendirir.

**Kullanım:**
```python
from regression_models import RegressionModels
regressor = RegressionModels(X_train, X_test, y_train_duration, y_test_duration)
regressor.train_and_evaluate_all()
```

### `clustering_models.py`
Kümeleme algoritmalarını uygular.

**Kullanım:**
```python
from clustering_models import ClusteringModels
clusterer = ClusteringModels(X_data)
optimal_k_info = clusterer.train_all()
```

### `results_analysis.py`
Tüm sonuçları analiz eder ve karşılaştırır.

**Kullanım:**
```python
from results_analysis import ResultsAnalyzer
analyzer = ResultsAnalyzer()
analyzer.set_classification_results(classification_results)
analyzer.print_all_results()
```

## 📝 Notlar

- Tüm Python dosyaları bağımsız olarak çalıştırılabilir (test amaçlı)
- Jupyter Notebook tüm analizi adım adım içerir
- Görselleştirmeler otomatik olarak `visualizations/` klasörüne kaydedilir
- Sonuç raporu `results_report.txt` dosyasına yazılır

## ✅ Proje Gereksinimleri

- [x] Veri ön işleme (gürültü, eksik değerler, normalizasyon, boyut azaltma)
- [x] En az 3 farklı diyagram ile görselleştirme
- [x] Sınıflandırma, regresyon, kümeleme yöntemleri
- [x] Farklı algoritmalar ile karşılaştırma
- [x] Uygun metriklerle değerlendirme
- [x] Sonuçların tartışılması ve çıkarımlar

## 👨‍💻 Geliştirici

Data Science Dersi Projesi - 2026

## 📄 Lisans

Bu proje eğitim amaçlıdır.
