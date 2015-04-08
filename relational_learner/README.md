# Relational Learner
A ROS package that uses qualitative spatio-temporal relations to learn and classify human trajectories as statistically novel or not. 


Prerequisites
-------------

- roscore
- mongodb_store
- strands_perception_people
- soma_trajectory
- qsrlib

Getting started (general steps)
-------------------------------
1. Start the ros core:

    ```
   $ roscore
    ```
2. Launch the ROS datacentre:

    ```
    $ roslaunch mongodb_store mongodb_store.launch db_path:= <path>
    $ roslaunch mongodb_store mongodb_store.launch db_path:=/home/strands/mongodb_store/bham_trajectory_store/
    ```
  where `path` specifies the path to your mongodb_store

3. Run people perception to publish detected trajectories:

    ```
    $ roslaunch ...
    ```

  Alternatively, you can run `soma_trajectory` and obtain test trajectories from mongodb store:
  
    ```
    $ rosrun soma_trajectory trajectory_query_service.py 
    ```
  
5. Run QSRLib Service:

    ```
    $ rosrun qsr_lib qsrlib_ros_server.py 
    ```

5. Run Episodes Service:

    ```
    $ rosrun relational_learner episode_server.py
    ```

6. Run Novelty Service:

    ```
    $ rosrun relational_learner novelty_server.py
    ```

7. Run Episodes Client: 
    ```
    $ rosrun relational_learner episodes_client.py
    ```

8. Run Novelty Client: 
    ```
    $ rosrun relational_learner novelty_client.py
    ```

Note:
-----
This package can be run offline by running `soma_trajectory` in step 3 instead of `people_perception`. In this case, step 7 becomes:
    ```
    $ rosrun relational_learner episodes_client_OT.py
    ```
This queries one region's trajectories from mongodb_store instead of subscribing to published trajectories. 


