#!/usr/bin/env python

"""Activity_Graph.py: File with Activity Graph Class."""

__author__      = "Krishna Dubba & Paul Duckworth"
__copyright__   = "Copyright 2015, University of Leeds"

import os
import multiprocessing
import time

from igraph import Graph
from itertools import combinations, permutations, product
from graph_hash import graph_hash
from interval_utils import get_valid_graphlets_from_activity_intervals, \
     get_temporal_neighbourhood_graphlets_from_activity_intervals, Episodes


def get_allen_relation(is1, ie1, is2, ie2):

    if is2-1 == ie1:
        return 'meets'
    elif is1-1 == ie2:
        return 'metby'
    
    elif is1 == is2 and ie1 == ie2:
        return 'equal'
    
    elif is2 > ie1:
        return 'before'
    elif is1 > ie2:
        return 'after'
    
    elif ie1 >= is2 and ie1 <= ie2 and is1 <= is2:
        return 'overlaps'
    elif ie2 >= is1 and ie2 <= ie1 and is2 <= is1:
        return 'overlapped_by' 
    elif is1 >= is2 and ie1 <= ie2:
        return 'during'
    elif is1 <= is2 and ie1 >= ie2:
        return 'contains'
    elif is1 == is2 and ie1 < ie2:        
        return 'starts'
    elif is1 == is2 and ie1 > ie2:
        return 'started_by'
    elif ie1 == ie2 and is2 < is1:
        return 'finishes'
    elif ie1 == ie2 and is2 > is1:
        return 'finished_by'  

    
def episodes_to_activity_graph(episodes_iter, params):
    """Converts a list of list of episodes into activity graphs
    """
    if not isinstance(episodes_iter[0], Episodes):
        return Activity_Graph(episodes_iter, params)
    else:
        activity_graphs = []
        for episode_list in episodes_iter:
            activity_graphs.append(Activity_Graph(episode_list.episodes, params))
        return activity_graphs


def get_graph_hash(graphs):
    if isinstance(graphs, Activity_Graph):
        return graph_hash(graphs.abstract_graph)
    
    return map(graph_hash, [ag.abstract_graph for ag in graphs])
  
  
def f(valid_eps, params):

    graphlet_hash_cnts={}
    valid_graphlets={}
    
    window, list_of_graphlets = valid_eps
    valid_graphlets[window]={}
    valid_graphlets_cnts = {}           

    for single_graphlet in list_of_graphlets:


        act_graphlets = episodes_to_activity_graph(single_graphlet, params)
        (ghash, g) = get_graph_hash(act_graphlets)
        
        if ghash in valid_graphlets_cnts:
            #This adds one to the graph hash (graphlet) count
            cnt = valid_graphlets_cnts[ghash]
            cnt += 1
            valid_graphlets_cnts[ghash] = cnt
        else:
            #This adds the graph hash to the graphlet list
            valid_graphlets_cnts[ghash] = 1
            
            #These are the actual iGraph graphlets:
            valid_graphlets[window][ghash] = g
            
    graphlet_hash_cnts[window] = valid_graphlets_cnts    
    return (graphlet_hash_cnts, valid_graphlets)


class Activity_Graph():
    '''
    Activity Graph class:
    Lower level is only nodes of type 'object'. Middle level nodes are only of 
    type 'spatial_relation'. Top level nodes are only of type 'temporal_relation'.
    Accepts input file which is a plain text file with each line an interaction of
    a pair of objects. 
    Header: object1, object1_type, object2, object2_type, spatial relation, start time, end time
    
    Example content:
    ----------------
    o1,mug,o2,hand,sur,3,7
    o1,mug,o3,head,con,4,9
    o2,hand,o3,head,dis,1,9
    
    OR
    
    Accepts a list of episodes where each episode is a tuple with above structure.
    '''
    def __init__(self, episodes, params, COLLAPSE_TEMPORAL_NODES=True):
        self.spatial_obj_edges  = []
        self.temp_spatial_edges = []
        self.episodes           = []

        self.valid_graphlets    = {}
        self.graphlet_hash_cnts = {}

        self.graph              = self.get_activity_graph(episodes, COLLAPSE_TEMPORAL_NODES)
        self.params = params

    @property    
    def abstract_graph(self):
        #abstract_graph = Graph.copy(self.graph)
        abstract_graph = self
        
        for o_node in self.object_nodes:
            #Activity Graph code:
            #Set the name of the object node equal to the type. 
            o_node['name']=o_node['obj_type']
            
            #to remove object node type and name:
            """o_node['obj_type'] = 'Unknown'
            o_node['name']='Unknown_object'
            """ 
        return abstract_graph    
    
    @property
    def object_nodes(self):
        # Get object nodes from graph
        object_nodes = []
        for node in self.graph.vs():
            if node['node_type'] == 'object':
                object_nodes.append(node)
        return object_nodes
    
    @property
    def abstract_object_nodes(self):
        # Get object nodes from abstract graph
        object_nodes = []
        for node in self.abstract_graph.vs():
            if node['node_type'] == 'object':
                object_nodes.append(node)
        return object_nodes
    
    @property
    def spatial_nodes(self):
        # Get spatial relation nodes from graph
        spatial_nodes = []
        for node in self.graph.vs():
            if node['node_type'] == 'spatial_relation':
                spatial_nodes.append(node)
        return spatial_nodes
        
    @property
    def temporal_nodes(self):
        # Get temporal relation nodes from graph
        temporal_nodes = []
        for node in self.graph.vs():
            if node['node_type'] == 'temporal_relation':
                temporal_nodes.append(node)
        return temporal_nodes


    def get_valid_graphlets(self, Force_Compute=False):
        if len(self.graphlet_hash_cnts) > 0 and not Force_Compute:
            return

        valid_graphlets_dict = get_valid_graphlets_from_activity_intervals(\
            self.episodes, self.params)      

        list_of_output=[]
        list_of_output.append(f(valid_graphlets_dict.items()[0], self.params))
                        
        #Extract the two dictionaries from the list output
        valid_graphlet_windowed={}
        valid_graphlets={}
        for (a, b) in list_of_output:
            #Keys = windows. Values = Dictionary of Hash value : Quantity in video clip 
            self.graphlet_hash_cnts[a.keys()[0]] = a.values()[0]
            #Keys = windows. Values = Dictionary of Hash value : Activity Graph instance
            self.valid_graphlets[b.keys()[0]] = b.values()[0]                   

        return



    def get_histogram(self, code_book):
        self.get_valid_graphlets()
        histograms_per_window = {}

        #for window in self.graphlet_hash_cnts.keys():    #If using multiple windows
        window = self.graphlet_hash_cnts.keys()[0]

        histogram = [0.0] * len(code_book)
        for bin_pos in range(len(code_book)):
            if code_book[bin_pos] in self.graphlet_hash_cnts[window]:
                histogram[bin_pos] += self.graphlet_hash_cnts[window][code_book[bin_pos]]

        #histograms_per_window[window] = histogram
        return histogram

    def _get_episodes_from_file(self, episodes_file):
        episodes      = []
        episodes_text = open(episodes_file)
        
        # Line structure example: o1,rh,o2,lh,DC,1,116
        for line in episodes_text:
            line   = line.strip()
            fields = line.split(',') 
            episodes.append((fields[0],fields[1],fields[2],fields[3],fields[4],int(fields[5]),int(fields[6])))
        return episodes    
    
    
    #returns the Start (E_s) and End (E_s) sets of a selection of episodes
    def get_E_set(self, objects, episodes_selection):
    
        objects = objects.values()
        start = {}
        end = {}
        E_s = []
        E_f = []

        for obj1, obj2 in combinations(objects, 2):
            added=0
            for epi in episodes_selection:
                if (epi[0] == obj1 and epi[1] == obj2) or (epi[0] == obj2 and epi[1] == obj1):
                    added=1
                    start[epi[3]] = epi
                    end[epi[4]] = epi
            if added == 1:
                st=start.keys()
                st.sort()
                E_s.append(start[st[0]])
                
                en=end.keys()
                en.sort()
                E_f.append(end[en[-1]])
                
        return (E_s, E_f)    
    
        
    def get_activity_graph(self, input_episodes, COLLAPSE_TEMPORAL_NODES):
        # Generate activity graph from file with object interaction information
        temporal_map = {'after': 'before',
                        'metby' : 'meets',
                        'overlapped_by': 'overlaps',
                        'started_by': 'starts',
                        'contains':'during',
                        'finished_by': 'finishes'
                        }
        
        if type(input_episodes) != list:
            # Episodes are given in a file
            episodes = self._get_episodes_from_file(input_episodes)
        else:
            episodes = input_episodes
            
        self.episodes = episodes
        
        objects = {}                        
        data    = []
        types   = []
        spatial_rels = {}
        vertex_count = 0
        graph = Graph(directed=True)
        
        for (o1, o1t, o2, o2t, rel, intv_start, intv_end) in episodes:

            # Add objects to the graph
            if o1 not in objects:
                graph.add_vertex(o1)
                objects[o1] = vertex_count
                graph.vs()[vertex_count]['obj_type'] = o1t
                graph.vs()[vertex_count]['node_type'] = 'object'
                vertex_count += 1
            if o2 not in objects:
                graph.add_vertex(o2)
                objects[o2] = vertex_count
                graph.vs()[vertex_count]['obj_type'] = o2t
                graph.vs()[vertex_count]['node_type'] = 'object'
                vertex_count += 1
            # Add spatial node to the graph    
            graph.add_vertex(rel)
            graph.vs()[vertex_count]['node_type'] = 'spatial_relation'
            # Add edges from spatial node to objects
            graph.add_edge(objects[o1], vertex_count)
            graph.add_edge(vertex_count, objects[o2])
            self.spatial_obj_edges.append((objects[o1], vertex_count))
            self.spatial_obj_edges.append((vertex_count, objects[o2]))
            data.append((objects[o1], objects[o2], vertex_count, intv_start, intv_end))
            vertex_count += 1
        
        (E_s, E_f) = self.get_E_set(objects, data)
        
        temporal_vertices = {}
        for ((o11, o12, rel1, s1, e1),(o21, o22, rel2, s2, e2)) in combinations(data,2):

            
            #No temporal relation between two starting episodes or between two final episodes
            if ((o11, o12, s1, e1) in E_s and (o21, o22, s2, e2) in E_s) or ((o11, o12, s1, e1) in E_f and (o21, o22, s2, e2) in E_f):
                continue
                         
            temporal_rel = get_allen_relation(s1, e1, s2, e2)
            # If temporal_rel is in temporal_map get its value otherwise keep it the same
            # If the edges are directed, then we need to change the direction of the edges
            # if we change change the temporal relation to its inverse
            temporal_rel_new = temporal_map.get(temporal_rel, temporal_rel)
            # Add temporal node to the graph
            if COLLAPSE_TEMPORAL_NODES and temporal_rel_new in temporal_vertices:
                temporal_rel_vertex_id = temporal_vertices[temporal_rel_new]
            else:
                graph.add_vertex(temporal_rel_new)
                graph.vs()[vertex_count]['node_type'] = 'temporal_relation'
                temporal_rel_vertex_id          = vertex_count
                temporal_vertices[temporal_rel_new] = vertex_count
                vertex_count += 1       
                
            if temporal_rel_new == temporal_rel:                           
                # Add edges from temporal node to the spatial nodes
                graph.add_edge(rel1, temporal_rel_vertex_id)
                graph.add_edge(temporal_rel_vertex_id, rel2)            
                self.temp_spatial_edges.append((rel1, temporal_rel_vertex_id))
                self.temp_spatial_edges.append((temporal_rel_vertex_id, rel2))
            else:
                # PD: if an inverse temporal relation has been used, switch the edges around
                graph.add_edge(temporal_rel_vertex_id, rel1)
                graph.add_edge(rel2, temporal_rel_vertex_id)            
                self.temp_spatial_edges.append((temporal_rel_vertex_id, rel1))
                self.temp_spatial_edges.append((rel2, temporal_rel_vertex_id)) 
        return graph        
   
     
    def graph2dot(self, out_dot_file, COLLAPSE_TEMPORAL_NODES=True):
        # Write the graph to dot file
        # Can generate a graph figure from this .dot file using the 'dot' command
        # dot -Tpng input.dot -o output.png
        dot_file = open(out_dot_file, 'w')
        dot_file.write('digraph activity_graph {\n')
        dot_file.write('    size = "45,45";\n')
        dot_file.write('    node [fontsize = "18", shape = "box", style="filled", fillcolor="aquamarine"];\n')
        dot_file.write('    ranksep=5;\n')
        # Create temporal nodes
        dot_file.write('    subgraph _1 {\n')   
        dot_file.write('    rank="source";\n')
        
        if COLLAPSE_TEMPORAL_NODES:
            temporal_names = {'before'  :0,
                              'meets'   :0,
                              'overlaps':0,
                              'starts'  :0,
                              'during'  :0,
                              'finishes':0,
                              'equal'   :0
                             }                        
            for tnode in self.temporal_nodes:
                temporal_names[tnode['name']] = tnode.index
            for tnode in temporal_names:
                dot_file.write('    %s [fillcolor="white", label="%s", shape=ellipse];\n' \
                               %(temporal_names[tnode], tnode))
        else:        
            for tnode in self.temporal_nodes:
                dot_file.write('    %s [fillcolor="white", label="%s", shape=ellipse];\n' %(tnode.index, tnode['name']))
                
        dot_file.write('}\n')
        
        # Create spatial nodes
        dot_file.write('    subgraph _2 {\n')   
        dot_file.write('    rank="same";\n')
        for rnode in self.spatial_nodes:
            dot_file.write('    %s [fillcolor="lightblue", label="%s"];\n' %(rnode.index, rnode['name']))
        dot_file.write('}\n')
        
        # Create object nodes
        dot_file.write('    subgraph _3 {\n')   
        dot_file.write('    rank="same";\n')        
        for onode in self.object_nodes:
            dot_file.write('%s [fillcolor="tan1", label="%s"];\n' %(onode.index, onode['name']))
        dot_file.write('}\n')
         
        # Create temporal to spatial edges
        if COLLAPSE_TEMPORAL_NODES:
            for (source, target) in self.temp_spatial_edges:
                if self.graph.vs[source]['node_type'] == 'temporal_relation':
                    source = temporal_names[self.graph.vs[source]['name']]
                elif self.graph.vs[target]['node_type'] == 'temporal_relation':
                    target = temporal_names[self.graph.vs[target]['name']]
                dot_file.write('%s -> %s [arrowhead = "normal", color="red"];\n' %(source, target))
        else:    
            for t_edge in self.temp_spatial_edges:
                dot_file.write('%s -> %s [arrowhead = "normal", color="red"];\n' %(t_edge[0], t_edge[1]))
            
        # Create spatial to object edges    
        for r_edge in self.spatial_obj_edges:
            dot_file.write('%s -> %s [arrowhead = "normal", color="red"];\n' %(r_edge[0], r_edge[1]))
        dot_file.write('}\n')
        dot_file.close()    
        

    def graph2hive(self, out_hive_file, temporal_rels, spatial_rels, obj_types,\
                   collapse_spatial_nodes=False, collapse_object_nodes=False):
        import numpy as np

        temporal_code = 0
        spatial_code  = 1
        obj_type_code = 2
        all_nodes = []
        nodes_map = {}
        node_count = 0
        temporal_names = {'before'  :"B",
                          'meets'   :"M",
                          'overlaps':"O",
                          'starts'  :"S",
                          'during'  :"D",
                          'finishes':"F",
                          'equal'   :"E"
                         }
                
        temporal_linspace = np.linspace(0.1, 0.9, num=len(temporal_rels))
        for i in xrange(len(temporal_rels)):
            rel_name = temporal_names[temporal_rels[i]]
            node_string = '{x: %d, y: %f, name:"%s"}' %(temporal_code, temporal_linspace[i], rel_name)
            all_nodes.append(node_string)
            nodes_map[temporal_rels[i]] = node_count
            node_count += 1
        
        if collapse_spatial_nodes:
            spatial_linspace  = np.linspace(0.1, 0.9, num=len(spatial_rels))
            for i in xrange(len(spatial_rels)):
                node_string = '{x: %d, y: %f, name:"%s"}' %(spatial_code, spatial_linspace[i], spatial_rels[i])
                all_nodes.append(node_string)
                nodes_map[spatial_rels[i]] = node_count
                node_count += 1
        else:
            spatial_linspace  = np.linspace(0.1, 0.9, num=len(self.spatial_nodes))
            for i in xrange(len(self.spatial_nodes)):
                spatial_node = self.spatial_nodes[i]
                node_string = '{x: %d, y: %f, name:"%s", node_id:"%d"}' %(spatial_code, spatial_linspace[i],\
                                                                          spatial_node['name'], spatial_node.index)
                all_nodes.append(node_string)
                nodes_map[spatial_node.index] = node_count
                node_count += 1
                
        if collapse_object_nodes:
            obj_type_linspace = np.linspace(0.1, 0.9, num=len(obj_types))
            for i in xrange(len(obj_types)):
                node_string = '{x: %d, y: %f, name:"%s"}' %(obj_type_code, obj_type_linspace[i], obj_types[i])
                all_nodes.append(node_string)
                nodes_map[obj_types[i]] = node_count
                node_count += 1 
        else:                
            obj_type_linspace = np.linspace(0.1, 0.9, num=len(self.object_nodes))
            for i in xrange(len(self.object_nodes)):
                obj_node = self.object_nodes[i]
                node_string = '{x: %d, y: %f, name:"%s", node_id:"%d"}' %(obj_type_code, obj_type_linspace[i],\
                                                                          obj_node['name'], obj_node.index)
                all_nodes.append(node_string)
                nodes_map[obj_node.index] = node_count
                node_count += 1
                
        temporal_edges = {}
        spatial_edges  = {}
        for edge in self.graph.es():
            snode = self.graph.vs[edge.source]
            tnode = self.graph.vs[edge.target]
            
            if snode['node_type'] == 'object' and collapse_object_nodes:
                source = nodes_map[snode['obj_type']]
            elif snode['node_type'] == 'temporal_relation' \
                 or (snode['node_type'] == 'spatial_relation' and collapse_spatial_nodes):
                source = nodes_map[snode['name']]
            else:
                source = nodes_map[snode.index]
                
            if tnode['node_type'] == 'object' and collapse_object_nodes:
                target = nodes_map[tnode['obj_type']]
            elif tnode['node_type'] == 'temporal_relation' \
                 or (tnode['node_type'] == 'spatial_relation' and collapse_spatial_nodes):
                target = nodes_map[tnode['name']]
            else:
                target = nodes_map[tnode.index]
                
            if snode['node_type'] == 'object' or tnode['node_type'] == 'object':    
                if (source,target) not in spatial_edges:
                    spatial_edges[(source,target)] = 0
                spatial_edges[(source,target)] += 1        
            else:
                if (source,target) not in temporal_edges:
                    temporal_edges[(source,target)] = 0                
                temporal_edges[(source,target)] += 1
                        
        spatial_max_edge_width  = max(spatial_edges.values())                
        temporal_max_edge_width = max(temporal_edges.values())
        max_edge_width = max(spatial_max_edge_width, temporal_max_edge_width)
        
        hive_file = open(out_hive_file, 'w')
        initial_string ='<!DOCTYPE html>\n\
                        <meta charset="utf-8">\n\
                        <style>\n\
                        \n\
                        .link {\n\
                          fill: none;\n\
                          stroke-width: 1.5px;\n\
                        }\n\
                        \n\
                        .axis, .node {\n\
                          stroke: #000;\n\
                          stroke-width: 2px;\n\
                        }\n\
                        \n\
                        </style>\n\
                        <body>\n\
                        <script src="http://d3js.org/d3.v3.min.js"></script>\n\
                        <script src="http://d3js.org/d3.hive.v0.min.js"></script>\n\
                        <script>\n\
                        \n\
                        var width = 960,\n\
                            height = 900,\n\
                            innerRadius = 40,\n\
                            outerRadius = 450;\n\
                        \n\
                        var angle  = d3.scale.ordinal().domain(d3.range(4)).rangePoints([0, 2 * Math.PI]),\n\
                            radius = d3.scale.linear().range([innerRadius, outerRadius]),\n\
                            thick  = d3.scale.linear().range([1, %d]),\n\
                            color  = d3.scale.category10().domain(d3.range(20));\n\
                        \n' % (max_edge_width)
        
        nodes_string = 'var nodes = [\n'
        for node in all_nodes:
            nodes_string += node + ',\n'
        else:
            nodes_string += '];\n'
            nodes_string += '\n'
 
        link_string = 'var links = [\n'
        for (source,target) in temporal_edges:    
            link_string += '{source: nodes[%d], target: nodes[%d], count:%d, weight:%d},\n' % (source,target,temporal_edges[(source,target)],1)       
            
        for (source,target) in spatial_edges:    
            link_string += '{source: nodes[%d], target: nodes[%d], count:%d, weight:%d},\n' % (source,target,spatial_edges[(source,target)],10)       
        else:
            link_string += '];\n'
            link_string += '\n'    

        hive_file.write(initial_string)
        hive_file.write(nodes_string)
        hive_file.write(link_string)
            
        hive_file.write('var svg = d3.select("body").append("svg")\n\
                            .attr("width", width)\n\
                            .attr("height", height)\n\
                          .append("g")\n\
                            .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");\n\
                        \n\
                        svg.selectAll(".axis")\n\
                            .data(d3.range(3))\n\
                          .enter().append("line")\n\
                            .attr("class", "axis")\n\
                            .attr("transform", function(d) { return "rotate(" + degrees(angle(d)) + ")"; })\n\
                            .attr("x1", radius.range()[0])\n\
                            .attr("x2", radius.range()[1]);\n\
                        \n\
                        svg.selectAll(".link")\n\
                            .data(links)\n\
                          .enter().append("path")\n\
                            .attr("class", "link")\n\
                            .attr("d", d3.hive.link()\n\
                            .angle(function(d) { return angle(d.x); })\n\
                            .radius(function(d) { return radius(d.y); }))\n\
                            .style("stroke", function(d) { return color(d.source.x); })\n\
                            .style("stroke-width", function(d) { return thick(d.count*d.weight/1800); });\n\
                        \n\
                        svg.selectAll(".node")\n\
                            .data(nodes)\n\
                          .enter().append("circle")\n\
                            .attr("class", "node")\n\
                            .attr("transform", function(d) { return "rotate(" + degrees(angle(d.x)) + ")"; })\n\
                            .attr("cx", function(d) { return radius(d.y); })\n\
                            .attr("r", 5)\n\
                            .style("fill", function(d) { return color(d.x); });\n\
                        \n\
                        svg.selectAll("text.label")\n\
                            .data(nodes)\n\
                          .enter().append("text")\n\
                            .attr("class", "label")\n\
                            .attr("fill", "black")\n\
                            .attr("transform", function(d) { return "rotate(" + degrees(angle(d.x)) + ")"; })\n\
                            .attr("dx", function(d) { return radius(d.y+0.02); })\n\
                            .text(function(d) {  return d.name;  });\n\
                        \n\
                        function degrees(radians) {\n\
                          return radians / Math.PI * 180 - 90;\n\
                        }\n\
                        \n\
                        </script>')    
        hive_file.close()
  
  
      
if __name__  == '__main__':
    import sys
    import cPickle as pickle
    import itertools
    from novelTrajectories.traj_data_reader import *

    date     = '__19_02_2015'
    base_data_dir = '/home/strands/STRANDS/'
    qsr_dir = os.path.join(base_data_dir, 'qsr_dump/')

    params = (None, 1, 3, 4)
    params_tag = map(str, params)
    params_tag = '_'.join(params_tag)
    tag = params_tag + date 
    print params
    print tag

    episodes_file = 'd6c54902-3259-5ff4-b1ca-9ed5132df53d__1__101'
    ep = Episodes(load_from_file="roi_12_episodes.p", dir=base_data_dir)
    ep2 = ep.all_episodes[episodes_file]
    episodes      = list(itertools.chain.from_iterable(ep2.values()))

    activity_graph = Activity_Graph(episodes, params)
    activity_graph.get_valid_graphlets()

    activity_graph.graph2dot('/tmp/TEST.dot', False)
    os.system('dot -Tpng /tmp/TEST.dot -o /tmp/TEST.png')
    print activity_graph.graph
    #act_graph.graph2hive('/tmp/act_hive_no_coll.html',temporal_rels, spatial_rels, obj_types)

    """
    print "hash counts = " + repr(activity_graph.graphlet_hash_cnts)
    print ""
    for hash in activity_graph.valid_graphlets[(1,101)]:
        for vertex in activity_graph.valid_graphlets[(1,101)][hash].object_nodes:
            print vertex['obj_type']
    """
 
