import sys
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from IPython.display import display


def read_data(dataset: pd.DataFrame, sep, names=None):
    df = pd.read_csv(dataset, sep=sep, names=names)
    df = pd.DataFrame(df)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    return df


def split_dataset(df, x, total_split):

    mask_train = int(df.shape[0] * (80  / 100))
    mask_test = int(df.shape[0] * (20  / 100))
    test_start = x * mask_test
    test_end = test_start + mask_test
    test_set = df[test_start:test_end]
    train_set = pd.concat([df[:test_start], df[test_end:]])
    '''test_start = x * mask
    test_end = test_start + mask
    test_set = df[test_start:test_end]
    train_set = pd.concat([df[:test_start], df[test_end:]])'''
    # print(test_set,'\n', train_set)

    return train_set, test_set



def euclidean_distance(row1, row2, col):
    #print(row1)
    distance = 0
    for c in col[:-1]:
        row1_val = getattr(row1, c)
        row2_val = getattr(row2, c)
        distance += (row1_val - row2_val) ** 2
    #print(distance)
    return sqrt(distance)


def get_relevant_points(dataset, points, dis_min):
    # print(points)
    # distance = [[-1 for x in range(n)]for y in range(len(dataset))]
    df = pd.concat([dataset, points], sort=False)
    df = df.drop_duplicates(keep=False)
    min_dist = dis_min
    min_dis_point = -1
    avg_dis = 0
    for cnt_df, i in df.iterrows():

        for cnt_points, j in points.iterrows():
            dis = euclidean_distance(i, j, dataset.columns.values.tolist())
            if dis < min_dist:
                min_dis_point = i.to_dict()
                min_dist = dis

    # print(min_dist, '\n', min_dis_point)
    _list = []
    _list.append(min_dis_point)
    min_dis_point = pd.DataFrame(_list)
    print(min_dis_point)


def get_avg_point(tl, n):
    df_list = [[] for y in range(n)]
    avg_points = [0 for y in range(n)]
    #print (df_list)

    for x in range(n):
        _list = []
        for l in tl[x]:
            _list.append(l.to_dict())
        avg_points[x] = pd.DataFrame(_list).mean()

    #print(avg_points)
    return avg_points


def get_grp_point(dataset, n, test_points, div = False):
    avg_dis = 0
    global test_list
    global divided_grps
    test_list1 = [[] for y in range(n)]
    for cnt_df, i in dataset.iterrows():
        min_dist = sys.maxsize
        min_idx = -1
        for cnt_points, j in enumerate(test_points.iterrows()):
            #print(cnt_points)
            dis = euclidean_distance(i, j[1], dataset.columns.values.tolist())
            if dis < min_dist:
                #min_dis_point = i.to_dict()
                min_dist = dis
                min_idx = cnt_points
        #print(min_idx)
        test_list1[min_idx].append(i)
    #print(test_list)
    divided_grps = test_list1
    if div:
        return
    avg_pts = get_avg_point(test_list1, n)
    _list = []
    for item in avg_pts:
        _list.append(item.to_dict())
    avg_pts_tt = pd.DataFrame(_list)

    for x in range(n):
        for index, row1 in test_points.iloc[[0]].iterrows():
            for index2, row2 in avg_pts_tt.iloc[[x]].iterrows():
                dis = euclidean_distance(row1, row2, dataset.columns.values.tolist())
                if dis < 1000:
                    test_list = avg_pts_tt
                    return
    get_grp_point(dataset, n, avg_pts_tt)

def divide_sets():
    pass

if __name__ == '__main__':
    df = read_data("heart_failure_clinical_records_dataset.csv", ',')
    # print(df)
    split_times = 5
    for x in range(split_times):
        train_set, test_set = split_dataset(df, x, split_times)
        death_set = train_set.loc[train_set['DEATH_EVENT'] == 1]
        num_of_grp = 3
        global test_list
        global divided_grps
        get_grp_point(death_set[:][num_of_grp:], num_of_grp, death_set[:][:num_of_grp])

        death_list = test_list
        #print(death_list.mean())
        d_l = death_list.mean()
        d_l = d_l.to_dict()
        survive_set = train_set.loc[train_set['DEATH_EVENT'] == 0]
        get_grp_point(survive_set[:][num_of_grp:], num_of_grp, survive_set[:][:num_of_grp])
        #print("Not Death")
        survive = test_list
        #print(survive.mean())
        s_l = survive.mean()

        s_l = s_l.to_dict()
        _list = []
        _list.append(s_l)
        _list.append(d_l)
        div_frame = pd.DataFrame(_list)

        death_list = death_list.to_dict()
        survive = survive.to_dict()
        _list1 = []
        _list1.append(s_l)
        _list1.append(d_l)
        div_all = pd.DataFrame(_list1)


        get_grp_point(test_set, 2, div_all, True)
        #print(divided_grps)
        d_grp = divided_grps
        _df_lst_1 = []
        for df_lists_1 in d_grp:
            _ll = []
            for df_s in df_lists_1:
                _ll.append(df_s.to_dict())
            _df_lst_1.append(pd.DataFrame(_ll))

        #print(_df_lst_1)
        for d_f_plt in _df_lst_1:
            #print(d_f_plt)
            d_f_plt.hist(column=['age', 'DEATH_EVENT'])
            plt.show()