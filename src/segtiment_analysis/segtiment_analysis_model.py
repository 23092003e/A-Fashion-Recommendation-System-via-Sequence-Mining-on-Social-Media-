import pandas as pd
import numpy as np
import re
import os
import emoji
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Tải thêm tài nguyên ngôn ngữ
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(filepath):
    df = pd.read_excel(filepath)
    print(f"Đã tải dữ liệu từ {filepath}. Kích thước: {df.shape}")
    return df

def explore_data(df):
    # Kiểm tra phân phối nhãn
    print("\nPhân phối nhãn:")
    label_counts = df['category'].value_counts()
    print(label_counts)
    
    # Kiểm tra độ dài bình luận
    df['comment_length'] = df['comments'].fillna('').apply(len)
    
    # Vẽ biểu đồ phân phối độ dài bình luận theo category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='category', y='comment_length', data=df)
    plt.title('Phân phối độ dài bình luận theo category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('comment_length_distribution.png')
    
    # Kiểm tra dữ liệu null
    print("\nSố lượng giá trị null trong các cột:")
    print(df.isnull().sum())
    
    return df

def clean_data(df):
    """Làm sạch dữ liệu"""
    # Loại bỏ các bản ghi trùng lặp
    df_cleaned = df.drop_duplicates(subset='comments', keep='first')
    
    # Xử lý các cột không cần thiết
    if 608 in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=608, axis=1)
    
    # Xử lý giá trị null
    df_cleaned['comments'] = df_cleaned['comments'].fillna('')
    
    # Chuyển đổi loại dữ liệu nếu cần
    df_cleaned['comments'] = df_cleaned['comments'].astype(str)
    
    print(f"Dữ liệu sau khi làm sạch: {df_cleaned.shape}")
    return df_cleaned

def preprocess_text(text):
    """Tiền xử lý văn bản nâng cao"""
    if not isinstance(text, str) or text is None or text == '':
        return ''
    
    # Chuyển emoji thành text
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Chuyển về chữ thường
    text = text.lower()
    
    # Thay thế các ký tự đặc biệt
    text = re.sub('::', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    text = re.sub(r'@[A-Za-z0-9]+', ' MENTION ', text)
    text = re.sub(r'#[A-Za-z0-9]+', ' HASHTAG ', text)
    
    # Xử lý cảm xúc phổ biến và từ viết tắt
    text = re.sub(r'(.)(\1{2,})', r'\1\1', text)  # Chuẩn hóa từ kéo dài (vd: 'soooo' -> 'soo')
    
    # Loại bỏ các ký tự không phải chữ cái hoặc số
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_and_lemmatize(text):
    """Tách từ và lemmatize"""
    if not isinstance(text, str) or text is None or text == '':
        return []
    
    # Tiền xử lý text
    text = preprocess_text(text)
    
    # Tách từ
    tokens = word_tokenize(text)
    
    # Loại bỏ stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def split_data_with_stratification(df):
    """Chia dữ liệu với stratification để đảm bảo phân phối nhãn đồng đều"""
    df2 = df[df['category'].notnull()]
    X = df2.comments
    y = df2.category
    
    # Giá trị test_size có thể điều chỉnh để tối ưu
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def handle_imbalanced_data(X_train, y_train):
    """Xử lý dữ liệu mất cân bằng bằng SMOTE"""
    # Tạo pipeline để chuyển đổi văn bản thành vector
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_lemmatize, max_features=5000, min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Áp dụng SMOTE để tạo dữ liệu cân bằng
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train)
    
    print(f"Phân phối nhãn sau khi áp dụng SMOTE: {pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled, vectorizer

def build_model():
    """Xây dựng mô hình với nhiều lựa chọn hơn"""
    # Pipeline với TfidfVectorizer thay vì CountVectorizer + TfidfTransformer
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(
            tokenizer=tokenize_and_lemmatize,
            max_features=5000,
            min_df=2,
            ngram_range=(1, 2),  # Thêm bigram
            use_idf=True,
            sublinear_tf=True  # Áp dụng sublinear scaling cho term-frequency
        )),
        ('clf', RandomForestClassifier(n_jobs=-1, class_weight='balanced'))
    ])
    
    # Tham số tìm kiếm
    params = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_features': [3000, 5000],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }
    
    # Sử dụng stratified k-fold để giữ phân phối nhãn đồng đều
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    return GridSearchCV(pipeline, param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)

def try_different_models(X_train, y_train, X_test, y_test):
    """Thử nghiệm nhiều mô hình khác nhau để tìm mô hình tốt nhất"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
        'SVM': SVC(kernel='linear', probability=True, class_weight='balanced')
    }
    
    # Tạo vectorizer cho dữ liệu
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_lemmatize,
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2)
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nHuấn luyện mô hình {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'vectorizer': vectorizer
        }
        
        print(f"Độ chính xác {name}: {accuracy}")
        print(f"Classification Report:\n{report}")
    
    # Tìm mô hình có độ chính xác cao nhất
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nMô hình tốt nhất: {best_model_name} với độ chính xác {results[best_model_name]['accuracy']}")
    
    return results[best_model_name]['model'], results[best_model_name]['vectorizer']

def evaluate_model(model, X_test, y_test, vectorizer=None):
    """Đánh giá mô hình và hiển thị kết quả chi tiết"""
    if vectorizer:
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
    else:
        y_pred = model.predict(X_test)
    
    # Tính độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    
    # Tạo báo cáo phân loại
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # Tạo ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    
    print('\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===')
    print(f'Độ chính xác: {accuracy:.4f}')
    print('\nBáo cáo phân loại:')
    print(report)
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return accuracy, report, cm

def analyze_errors(X_test, y_test, y_pred):
    """Phân tích các lỗi dự đoán để hiểu rõ về hạn chế của mô hình"""
    # Tạo DataFrame chứa các dự đoán
    error_df = pd.DataFrame({
        'text': X_test,
        'actual': y_test,
        'predicted': y_pred,
        'correct': y_test == y_pred
    })
    
    # Lọc ra các dự đoán sai
    errors = error_df[~error_df['correct']]
    
    print(f"\nSố lượng dự đoán sai: {len(errors)} trên tổng số {len(error_df)} mẫu thử ({len(errors)/len(error_df)*100:.2f}%)")
    
    # Phân tích lỗi theo từng nhãn
    for category in np.unique(y_test):
        category_errors = errors[errors['actual'] == category]
        if len(category_errors) > 0:
            print(f"\nLỗi khi nhãn thực tế là '{category}': {len(category_errors)} trên tổng số {len(error_df[error_df['actual'] == category])} mẫu")
            print(f"Phân bố nhãn dự đoán sai: {category_errors['predicted'].value_counts().to_dict()}")
    
    # Lưu các mẫu lỗi để phân tích thêm
    errors.to_csv('prediction_errors.csv', index=False)
    
    return errors

def save_model(model, vectorizer, model_filename='sentiment_analysis_model.joblib', vectorizer_filename='vectorizer.joblib'):
    """Lưu mô hình và vectorizer để sử dụng sau này"""
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    print(f"Đã lưu mô hình tại: {model_filename}")
    print(f"Đã lưu vectorizer tại: {vectorizer_filename}")

def predict_unlabelled_data(model, vectorizer, file_path):
    """Dự đoán nhãn cho dữ liệu chưa gán nhãn"""
    # Đọc dữ liệu
    df_comments = pd.read_csv(file_path)
    
    # Điền missing values
    df_comments['comments'] = df_comments['comments'].fillna('No comment')
    
    # Dự đoán nhãn
    comments_vector = vectorizer.transform(df_comments['comments'])
    df_comments['category'] = model.predict(comments_vector)
    
    # Tìm độ tin cậy của dự đoán
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(comments_vector)
        df_comments['prediction_confidence'] = np.max(probas, axis=1)
    
    # Lưu kết quả
    df_comments.to_csv(file_path, sep=';', index=False)
    print(f"Đã dự đoán nhãn cho {len(df_comments)} bình luận và lưu tại: {file_path}")
    
    # Phân tích phân bố nhãn dự đoán
    print("\nPhân bố nhãn dự đoán:")
    print(df_comments['category'].value_counts())

def main():
    # Đường dẫn dữ liệu
    label_path = r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\data\labelled_comments_train.xlsx"
    
    # 1. Tải và khám phá dữ liệu
    df = load_data(label_path)
    df = explore_data(df)
    
    # 2. Làm sạch dữ liệu
    df = clean_data(df)
    
    # 3. Chia dữ liệu với stratification
    X_train, X_test, y_train, y_test = split_data_with_stratification(df)
    
    # 4. So sánh các mô hình khác nhau
    best_model, vectorizer = try_different_models(X_train, y_train, X_test, y_test)
    
    # 5. Đánh giá mô hình chọn
    y_pred = best_model.predict(vectorizer.transform(X_test))
    accuracy, report, cm = evaluate_model(best_model, X_test, y_test, vectorizer)
    
    # 6. Phân tích lỗi
    errors = analyze_errors(X_test, y_test, y_pred)
    
    # 7. Lưu mô hình
    models_dir = r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\models"
    model_path = os.path.join(models_dir, "sentiment_analysis_model.joblib")
    vectorizer_path = os.path.join(models_dir, "vectorizer.joblib")
    save_model(best_model, vectorizer, model_path, vectorizer_path)
    
    # 8. Dự đoán dữ liệu chưa gán nhãn
    unlabelled_path = r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\data\processed\posts_comments.csv"
    predict_unlabelled_data(best_model, vectorizer, unlabelled_path)

if __name__ == "__main__":
    main()