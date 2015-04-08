#!/usr/bin/env python
import rospy
import sys, os, getpass
import ConfigParser


def get_path():
    user = getpass.getuser()
    data_dir = os.path.join('/home/' + user + '/STRANDS/')
    check_dir(data_dir)

    path = os.path.dirname(os.path.realpath(__file__))
    if path.endswith("/src/relational_learner"): 
        config_path = path.replace("/src/relational_learner", "/config.ini") 

    config_parser = ConfigParser.SafeConfigParser()
    print(config_parser.read(config_path))

    if len(config_parser.read(config_path)) == 0:
        raise ValueError("Config file not found, please provide a config.ini file as described in the documentation")

    return data_dir, config_path


def get_qsr_config(config_path):
    config_parser = ConfigParser.SafeConfigParser()
    print(config_parser.read(config_path))
    config_section = "soma" 
    try:
        map = config_parser.get(config_section, "soma_map")
        config = config_parser.get(config_section, "soma_config")
    except ConfigParser.NoOptionError:
         raise  
    return (map, config)


def check_dir(directory):
    if not os.path.isdir(directory):
        os.system('mkdir -p ' + directory)
    return


def get_learning_config():
    data_dir, config_path = get_path()

    qsr = os.path.join(data_dir, 'qsr_dump/')
    eps = os.path.join(data_dir, 'episode_dump/')
    graphs = os.path.join(data_dir, 'AG_graphs/')
    learning_area = os.path.join(data_dir, 'learning/')

    check_dir(qsr)
    check_dir(eps)
    check_dir(graphs)
    check_dir(learning_area)

    directories = (data_dir, qsr, eps, graphs, learning_area)

    config_parser = ConfigParser.SafeConfigParser()
    print(config_parser.read(config_path))
    config_section = "activity_graph_options"
    try:
        input_data={}
        date = config_parser.get(config_section, "date")
        input_data['MAX_ROWS'] = config_parser.get(config_section, "MAX_ROWS")
        input_data['MIN_ROWS'] = config_parser.get(config_section, "MIN_ROWS")
        input_data['MAX_EPI']  = config_parser.get(config_section, "MAX_EPI")
        input_data['num_cores'] = config_parser.get(config_section, "num_cores")

    except ConfigParser.NoOptionError:
        raise    

    return (directories, config_path, input_data, date)



if __name__ == "__main__":
    (data_dir, config_path, input_data) =  get_learning_config()
    print data_dir

    (map, config) = get_qsr_config(config_path)
    print map, config
    
