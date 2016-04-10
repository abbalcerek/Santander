from src.utils import data_path


def make_submission(file_name, test_id, prediction):
    with open(data_path(file_name), 'w') as f:
        f.write("ID,TARGET\n")
        for id, pred in zip(test_id, prediction):
            if hasattr(pred, '__iter__'): pred = pred[1]
            f.write('{},{}\n'.format(str(id), str(pred)))
