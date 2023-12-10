import tensorflow as tf
import numpy as np
import pandas as pd
import random
import streamlit as st

def restore_model_ckpt(user_id, dataset, epoch, max_len, item_len):
    sess = tf.Session()

    # load model structure
    saver = tf.train.import_meta_graph('./' + dataset + '_model_' + epoch + '/' + dataset + '_model.meta')  

     # restore model variables
    saver.restore(sess, tf.train.latest_checkpoint('./'+ dataset + '_model_' + epoch)) 
    
    # get placeholder variable
    is_training = sess.graph.get_tensor_by_name('Placeholder:0')
    u = sess.graph.get_tensor_by_name('Placeholder_1:0')
    input_seq = sess.graph.get_tensor_by_name('Placeholder_2:0')
    test_item = sess.graph.get_tensor_by_name('Placeholder_5:0')

    # get computational operator
    op = sess.graph.get_tensor_by_name('Reshape_7:0')
    op = op[:, -1, :]

    # parse processed data file
    item_list = []
    with open(dataset+'.txt') as file:
      for lstr in file:
        l = lstr.split(" ")
        uid = l[0]
        mid = l[1]

        if uid != str(user_id):
           continue
        item_list.append(int(mid))

    input_holder = [0] * max_len
    count = max_len
    for i in reversed(item_list[:-1]):
       if count < 1:
          break
       input_holder[count - 1] = i
       count = count - 1

    # Recent movie item
    # np.array([   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   83,
    #     150,  199,  286,  527,  561, 2117])
    test_item_temp = [item_list[len(item_list) - 1]] + random.sample([ele for ele in range(1, item_len) if ele != item_list[len(item_list) - 1]], 100)

    # Random moive item
    # [2283, 1435, 5765, 583, 1758, 909, 5324, 4922, 4940, 3653, 3778, 1084, 4505, 7195, 5090, 3022, 847, 2737, 557, 7097, 7251, 1805, 7171, 4460, 5771, 809, 6921, 6548, 5894, 4616, 834, 621, 476, 6282, 4832, 1261, 66, 4281, 3606, 1475, 5009, 6323, 4566, 7107, 3799, 6760, 767, 700, 4678, 5239, 5669, 6176, 5994, 6531, 1976, 2656, 878, 6369, 614, 3007, 491, 4488, 6815, 4506, 6121, 4030, 5397, 4100, 3449, 3889, 3592, 2800, 3823, 4680, 4319, 1219, 4327, 439, 4490, 884, 7066, 5815, 2475, 4489, 6412, 1968, 4912, 6227, 2944, 5716, 4866, 6349, 3365, 5476, 240, 6317, 7086, 3345, 4202, 893, 3027]
    
    # predict with selcted tensor input and operations
    predictions = -sess.run(op, {is_training: False, u: [user_id], input_seq:[input_holder], test_item:test_item_temp})

    predictions = predictions[0]
    rank = predictions.argsort().argsort()[0]
    top5 = [-1] * 5
    rank_list =  predictions.argsort().argsort()
    
    rank_c = 0
    for i in rank_list:
        if i < 5:
            top5[i] = test_item_temp[rank_c]
            
        rank_c = rank_c + 1

    sess.close()
    return item_list, item_list[len(item_list) - 1], top5, rank

def process(uid):
    item_list, target_movie_id, test_item_temp, rank = restore_model_ckpt(user_id=uid, dataset="ml1m", epoch='2000', max_len=200, item_len=3417)

    recent_movie_and_genere = []
    random_movie_and_genere = []

    # read movies dat file from movielens 1M
    mnames=['movie_id','title','genres']
    movies=pd.read_table('movies.dat',sep='::',header=None,names=mnames,engine='python')
    for i in item_list[:-1]:
        recent_movie_and_genere.append(movies["title"].iloc[i - 1] + "::" + movies["genres"].iloc[i - 1])

    target_movie_and_genere = movies["title"].iloc[target_movie_id - 1] + " with " + movies["genres"].iloc[target_movie_id - 1]
    
    for i in test_item_temp:
        random_movie_and_genere.append(movies["title"].iloc[i - 1] + "--" + movies["genres"].iloc[i - 1])
    return rank, target_movie_and_genere, list(reversed(recent_movie_and_genere)), random_movie_and_genere


# code for streamlit

# read user dat file from movielens 1M
users_cols=["UserID","Gender","Age","Occupation","Zip-code"]
users=pd.read_table('users.dat',sep='::',header=None,names=users_cols,engine='python')

st.markdown(
    f"""
        # MovieLens Test #
    """
)
uid = st.text_input("Enter the user ID")

if uid and uid != "":

    st.write("User info: gender is " + users["Gender"].iloc[int(uid) - 1] + ", age is " + str(users["Age"].iloc[int(uid) - 1]))

    rank, target_movie_and_genere, recent_movie_and_genere, random_movie_and_genere = process(uid)
    readble_rank = rank + 1
    st.write("The next movie (ground truth) should be " + target_movie_and_genere)
    st.write(f"The recommended ranking is No. :red[{readble_rank}] within the random list!")

    cates = dict()
    for i in recent_movie_and_genere:
        line = i.split("::")
        genres = line[1]
        genres_list = genres.split("|")
        for g in genres_list:
            if cates.get(g) is not None:
                cates[g] = cates[g] + 1
            else:
                cates[g] = 1

    cate_name = []
    cate_amount = []
    for k, v in cates.items():
        cate_name.append(k)
        cate_amount.append(v)
    bar_data = {"categories":cate_name, "amount":cate_amount}
    bar_df = pd.DataFrame(bar_data)
    bar_df = bar_df.set_index("categories")

    col1, col2 = st.columns(2)

    with col1:
        if recent_movie_and_genere != "":
            st.markdown(
            f"""
                    ### Most recent movie list ###
            """
            )
            st.write(recent_movie_and_genere)

    with col2:
        if random_movie_and_genere != "":
            st.markdown(
            f"""
                    ### Top 5 recommended movie list ###
            """
            )
            st.write(random_movie_and_genere)

    st.markdown(
    f"""
            ### History Distribution by type ###
    """
    )
    st.bar_chart(bar_df)
