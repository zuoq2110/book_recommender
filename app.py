from flask import Flask,render_template,request
import pickle
import numpy as np
import joblib

# Load all models and data
popular_df = pickle.load(open('popular_new.pkl','rb'))
pt = pickle.load(open('pt_new.pkl','rb'))
books = pickle.load(open('books_new.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores_new.pkl','rb'))

# Load ML models
svd_model = joblib.load('svd_model.pkl')
ml_data_filtered = pickle.load(open('ml_data_filtered.pkl', 'rb'))
trainset = pickle.load(open('trainset.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    # Debug: Print column names to check what's available
    print("Popular DataFrame columns:", popular_df.columns.tolist())
    
    return render_template('index.html',
                           book_name = list(popular_df['book_title'].values),
                           author=list(popular_df['authors'].values),
                           image=list(popular_df['cover_link'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/user_recommend')
def user_recommend_ui():
    # Get list of available users for the dropdown
    available_users = sorted(ml_data_filtered['customer_id'].unique())
    return render_template('user_recommend.html', users=available_users)

@app.route('/user_recommend_books', methods=['POST'])
def user_recommend():
    user_id = request.form.get('user_id')
    
    try:
        user_id = int(user_id)
        
        # Check if user exists in ML data
        if user_id not in ml_data_filtered['customer_id'].values:
            return render_template('user_recommend.html', 
                                 users=sorted(ml_data_filtered['customer_id'].unique()),
                                 error=f"User ID {user_id} không có trong hệ thống.")
        
        # Get ML-based recommendations
        recommendations = ml_recommend_for_user(user_id, n_recommendations=5)
        
        # Get user history for display
        user_history = ml_data_filtered[ml_data_filtered['customer_id'] == user_id].sort_values('rating', ascending=False)
        history_data = []
        for _, book in user_history.head(5).iterrows():
            history_data.append({
                'title': book['book_title'],
                'rating': book['rating']
            })
        
        return render_template('user_recommend.html', 
                             users=sorted(ml_data_filtered['customer_id'].unique()),
                             recommendations=recommendations,
                             user_history=history_data,
                             user_id=user_id)
        
    except ValueError:
        return render_template('user_recommend.html',
                             users=sorted(ml_data_filtered['customer_id'].unique()),
                             error="Vui lòng nhập User ID hợp lệ.")
    except Exception as e:
        return render_template('user_recommend.html',
                             users=sorted(ml_data_filtered['customer_id'].unique()),
                             error=f"Có lỗi xảy ra: {str(e)}")

def ml_recommend_for_user(user_id, n_recommendations=5):
    """
    Gợi ý sách cho user dựa trên Matrix Factorization (SVD)
    """
    # Lấy tất cả sách trong hệ thống
    all_books = ml_data_filtered['book_title'].unique()
    
    # Lấy sách mà user đã đánh giá
    user_rated_books = ml_data_filtered[ml_data_filtered['customer_id'] == user_id]['book_title'].unique()
    
    # Tìm sách chưa được đánh giá
    unrated_books = [book for book in all_books if book not in user_rated_books]
    
    # Dự đoán rating cho các sách chưa đánh giá
    predictions = []
    for book in unrated_books:
        pred = svd_model.predict(user_id, book)
        predictions.append((book, pred.est))
    
    # Sắp xếp theo predicted rating giảm dần
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy top N recommendations
    top_recommendations = predictions[:n_recommendations]
    
    # Format kết quả
    recommendations = []
    for book_title, predicted_rating in top_recommendations:
        book_info = books[books['title'] == book_title]
        if not book_info.empty:
            book_info = book_info.iloc[0]
            recommendations.append({
                'title': book_title,
                'author': book_info['authors'],
                'predicted_rating': round(predicted_rating, 2),
                'cover_link': book_info['cover_link']
            })
    
    return recommendations

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    
    # Check if book exists in pivot table
    if user_input not in pt.index:
        # Return error message or empty recommendations
        print(f"Book '{user_input}' not found in recommendation system")
        return render_template('recommend.html', 
                             data=[], 
                             error=f"Sách '{user_input}' không có trong hệ thống gợi ý. Vui lòng thử tên sách khác.")
    
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['title'] == pt.index[i[0]]]
        if not temp_df.empty:
            item.extend(list(temp_df.drop_duplicates('title')['title'].values))
            item.extend(list(temp_df.drop_duplicates('title')['authors'].values))
            item.extend(list(temp_df.drop_duplicates('title')['cover_link'].values))
            data.append(item)

    print(f"Recommendations for '{user_input}':", data)

    return render_template('recommend.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)