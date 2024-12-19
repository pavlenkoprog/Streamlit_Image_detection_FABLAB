import os
dir_dict = {}

def get_dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
            except FileNotFoundError:
                print(f"File not found: {fp}, skipping...")
    return total_size / (1024 ** 3)  # размер в гигабайтах

dir_number_fin = 0
dir_number = 0

def print_dir_tree(start_path='.', max_depth=None):
    global dir_number_fin, dir_number

    start_path = start_path.rstrip(os.path.sep)
    assert os.path.isdir(start_path)
    num_sep = start_path.count(os.path.sep)

    for dirpath, dirnames, filenames in os.walk(start_path):
        num_sep_this = dirpath.count(os.path.sep)
        if max_depth is not None and num_sep + max_depth < num_sep_this:
            del dirnames[:]
            continue
        dir_number_fin = dir_number_fin + 1
    print(dir_number_fin)

    for dirpath, dirnames, filenames in os.walk(start_path):
        num_sep_this = dirpath.count(os.path.sep)
        if max_depth is not None and num_sep + max_depth < num_sep_this:
            del dirnames[:]
            continue
        size = get_dir_size(dirpath)
        dir_number = dir_number + 1
        print(f"Directory: {dirpath}, Size: {size:.2f} GB")
        print(f"{dir_number/dir_number_fin * 100}%")
        # if size > 5:
        #     print("^ BIG FOLDER ^")
        dir_dict[dirpath] = size

    print("\n===========SORTED===========\n")
    sorted_dictionary = dict(sorted(dir_dict.items(), key=lambda item: item[1], reverse=True))

    # print("{" + "\n".join("{!r}: {:.2f},".format(k, v) for k, v in output.items()) + "}")
    # print(output)

    for key in list(sorted_dictionary)[:10]:
        print(f"{key}: {sorted_dictionary[key]:.2f}")



print_dir_tree("/home/dmitriy/Downloads", 3)  # замените на путь к вашей директории
