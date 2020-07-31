from util import *
from sklearn.linear_model import LogisticRegression as lg
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import GradientBoostingClassifier as gdb
from sklearn.ensemble import VotingClassifier as vote
import _pickle as pk
import sys

if __name__ == '__main__':
    x_train_path = sys.argv[3]
    y_train_path = sys.argv[4]
    test_path = sys.argv[5]
    output_path = sys.argv[6]

    x_train = pd.read_csv(x_train_path, header=0).to_numpy()
    y_train = pd.read_csv(y_train_path, header=None).to_numpy()
    x_test = pd.read_csv(test_path, header=0).to_numpy()

    scal_x_train = rescaling(x_train)
    scal_x_test = rescaling(x_test)

    is_train = False

    if is_train:
        with open('./model/v_model.pkl', 'rb') as f:
            v_model = pk.load(f)
    else:
        log_model = lg(solver='saga', C=10, verbose=1, n_jobs=-1, max_iter=100, penalty='elasticnet', l1_ratio=0.5)
        forest_model = rf(n_estimators=1000, max_depth=24, max_leaf_nodes=1000, n_jobs=-1, verbose=1, oob_score=True)
        gdb_model = gdb(n_estimators=1000, verbose=1, n_iter_no_change=10, validation_fraction=0.2, tol=1e-4)

        v_model = vote([('log', log_model), ('forest', forest_model), ('gdb', gdb_model)], voting='soft', n_jobs=-1, weights=[1, 3, 3])
        v_model.fit(scal_x_train, y_train.reshape(-1))
#        with open('./model/v_model.pkl', 'wb') as f:
#            pk.dump(v_model, f)
    pred_y = v_model.predict(scal_x_train).reshape(-1, 1)
    print('training accuracy: %.4f' % (acc(pred_y, y_train)))

    output_y = v_model.predict(scal_x_test).reshape(-1, 1)
    output(output_y, output_path)
