import os
def get_project_root() -> str:
    current_file = os.path.abspath(__file__)#当前文件的绝对路径

    current_dir = os.path.dirname(current_file)# 当前文件的目录

    project_root = os.path.dirname(current_dir)# 项目的根目录

    return project_root

def get_abs_path(relative_path:str) -> str:
    """

    :param relative_path: 相对路径
    :return: 绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)

if __name__ == '__main__':
    print(get_abs_path('config/config.txt'))