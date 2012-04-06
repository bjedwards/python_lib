import networkx as nx
import gl_windows as glw
import zmq

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

import warnings

try:
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    color_import = True
except:
    warnings.showwarning('Cannot import matplotlib.colors, or matplotlib.cm. All colors will need to be tuples', ImportWarning,'./nx_opengl.py',14)
    color_import = False

from collections import defaultdict
from itertools import repeat
    
class NetworkXGLWindow(object):
    
    def __init__(self,
                 G,
                 pos=None,
                 nodelist=None,
                 node_size=20,
                 node_color=(1.,0.,0.,1.),
                 node_cmap=cm.jet,
                 node_border_size=3,
                 node_border_color=(0.,0.,0.,1.),
                 node_border_cmap=cm.jet,
                 edgelist=None,
                 edge_color=(0.,0.,0.,0.2),
                 edge_cmap=cm.jet,
                 edge_style='-',
                 edge_thickness=1.0,
                 edge_self_loop_radius = .15,
                 with_arrows=True,
                 arrow_size=20,
                 with_node_labels=True,
                 node_label_font='Helvetica',
                 node_font_size=10,
                 node_label_color=(0.,0.,0.,1.),
                 node_label_cmap=cm.jet,
                 with_edge_labels=False,
                 edge_label_font='Helvetica',
                 edge_font_size=10,
                 edge_label_color=(0.,0.,0.,1.),
                 edge_label_cmap=cm.jet,
                 gl_window_title="NetworkX Graph",
                 gl_refresh=1,
                 gl_background_color=(1.,1.,1.,1.),
                 gl_width=800,
                 gl_height=800,
                 save_file='./',
                 save_file_type=None,
                 rescale_pos=True,
                 dim=3):
        data = {}
        
        if pos is None:
            pos=nx.drawing.spring_layout(G,dim=dim)
        if rescale_pos:
            max_pos = {}
            min_pos = {}
            for d in range(dim):
                max_pos[d] = float(max([pos[n][d] for n in pos]))
                min_pos[d] = float(min([pos[n][d] for n in pos]))

            for n in pos:
                pos[n] = tuple([1.7*((pos[n][d]-min_pos[d])/(max_pos[d]-min_pos[d])-.5) for d in range(dim)])
        data['pos'] = pos
        
        if nodelist is None:
            data['nodelist'] = G.nodes()
        else:
            data['nodelist'] = nodelist

        data['node_size'] = _parse_numeric_value(node_size,
                                                 data['nodelist'],
                                                 "Node Size")
        data['node_color'] = _parse_color_value(node_color,
                                                data['nodelist'],
                                                node_cmap,
                                                "Node Color")
        data['node_border_size'] = _parse_numeric_value(node_border_size,
                                                        data['nodelist'],
                                                        "Node Border Size")
        data['node_border_color'] = _parse_color_value(node_border_color,
                                                       data['nodelist'],
                                                       node_border_cmap,
                                                       "Node Border Color")
        if edgelist is None:
            if G.is_multigraph():
                data['edgelist'] = list(set(G.edges(keys=True)))
                self_loops = G.selfloop_edges(keys=True)
                data['multi_edge_ctrl_points'] = _calc_multi_edge_ctrl_points(G,
                                                                              pos,
                                                                              data['edgelist'],
                                                                              edge_self_loop_radius)
            else:
                data['edgelist'] = list(set(G.edges()))
                self_loops = G.selfloop_edges()
                data['multi_edge_ctrl_points'] = {}
        else:
            data['edgelist'] = edgelist

        data['self_loop_ctrl_points'] = _calc_self_loop_ctrl_points(G,
                                                                    pos,
                                                                    self_loops,
                                                                    edge_self_loop_radius)
            

        data['edge_color'] = _parse_color_value(edge_color,
                                                data['edgelist'],
                                                edge_cmap,
                                                "Edge Color")
        
        data['edge_thickness'] = _parse_numeric_value(edge_thickness,
                                                      data['edgelist'],
                                                      "Edge Thickness")
        
        data['edge_style']= _parse_line_stipple(edge_style,
                                         data['edgelist'],
                                         "Edge Style")

    
            
        data['with_node_labels'] = with_node_labels
        data['node_labels'] = dict(zip(data['nodelist'],map(str,data['nodelist'])))


        data['node_label_colors'] = _parse_color_value(node_label_color,
                                                       data['nodelist'],
                                                       node_label_cmap,
                                                       "Node Label Color")
        data['node_label_font'] = _parse_font_type_size(node_label_font,
                                                        node_font_size,
                                                        data['nodelist'],
                                                        "Node font and Size")
        
        data['with_edge_labels'] = with_edge_labels
        data['edge_labels'] = _calc_edge_labels(G,data['edgelist'])
        data['edge_label_pos'] = _calc_edge_label_pos(G,
                                                      pos,
                                                      data['edgelist'],
                                                      edge_self_loop_radius,
                                                      data['multi_edge_ctrl_points']) 
        data['edge_label_colors'] = _parse_color_value(edge_label_color,
                                                       data['edgelist'],
                                                       edge_label_cmap,
                                                       "Edge Label Color")
        data['edge_label_font'] = _parse_font_type_size(edge_label_font,
                                                        edge_font_size,
                                                        data['edgelist'],
                                                        "Edge font and Size")

        if not G.is_directed():
            data['with_arrows']=False
        else:
            data['with_arrows']=with_arrows

        if data['with_arrows']:
            data['arrow_size'] = _parse_numeric_value(arrow_size,
                                                      data['edgelist'],
                                                      "Arrow Size")
            data['arrow_points'] = _calc_arrow_points(pos,
                                                      data['edgelist'],
                                                      data['arrow_size'],
                                                      data['self_loop_ctrl_points'],
                                                      data['multi_edge_ctrl_points'],
                                                      gl_width,
                                                      gl_height)
        data['_dim'] = dim

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        p = self.socket.bind_to_random_port('tcp://127.0.0.1')
        data['_port'] = p

        self.gl_window = glw.gl_window(dim=dim,
                                       background_color=gl_background_color,
                                       refresh=gl_refresh,
                                       height=gl_width,
                                       width=gl_height,
                                       title=gl_window_title,
                                       comm_thread=_comm_thread,
                                       save_path=save_file,
                                       save_file_type=save_file_type,
                                       render_func=_graph_render_func,
                                       **data)


        

        
    def set_property(self,prop,k,v):
        self.socket.send_pyobj(prop)
        self.socket.send_pyobj(k)
        self.socket.send_pyobj(v)
        if prop=='exit':
            self.socket.close()
            self.context.term()


def _parse_numeric_value(x,l,description):
    #Takes a numeric value or list, sees makes a list of it repeated or
    # determines if it is long enough
    if type(x) is int or type(x) is float:
        return defaultdict(repeat(x).next)
    elif type(x) is dict:
        for n in l:
            if n not in x:
                raise RuntimeError("%s not present in "%n + "description")
        return x
    else:
        if not len(x)==len(l):
            raise RuntimeError("Dimensions of " + description + " do not match")
        else:
            return dict(zip(l,x))

def _parse_color_value(c,l,cmap,description):
    # Converts any number of color values into a list of RGBA tuples
    tc = type(c)
    if tc is tuple or tc is str:
        return defaultdict(repeat(colors.colorConverter.to_rgba(c)).next)
    elif tc is float:
        return defaultdict(repeat(cmap(c)).next)
    elif tc is dict:
        d = {}
        for x in c:
            tc = type(c[x])
            if tc is tuple or tc is str:
                d[x] = colors.colorConverter.to_rgba(c[x])
            elif tc is float or tc is int:
                c_max = float(max(c.values()))
                c_min = float(min(c.values()))
                d[x] = cmap((c[x]-c_min)/(c_max-c_min))
            else:
                d[x] = c[x]
        return d
    else:
        if not len(c)==len(l):
            raise RuntimeError("Dimensions of " + description + " do not match")
        else:
            tc = type(c[0])
            if tc is tuple or tc is str:
                return dict(zip(l,map(colors.colorConverter.to_rgba,c)))
            elif tc is float or tc is int:
                c_max = float(max(c))
                c_min = float(min(c))
                return dict(zip(l,map(cmap,[(ci - c_min)/(c_max-c_min) for ci in c])))
            else:
                return dict(zip(l,c))

def _parse_line_stipple(s,l,description):
    # Stipple parsing for openGL
    def string_match(s):
        if s == '-':
            return (1,0xFFFF)
        elif s == '--':
            return (1,0x000F)
        elif s == ':':
            return (1,0x0101)
        elif s == '-.':
            return (1,0xC204)
        else:
            raise RuntimeError("Cound not parse string for edge type")
    if type(s) is str:
        return defaultdict(repeat(string_match(s)).next)
    elif type(s) is dict:
        d = {}
        for x in s:
            d[x] = string_match(s[x])
        return d
    else:
        if not len(s) == len(l):
            raise RuntimeError("Dimensions of " + description + " do not match")
        else:
            if type(s[0]) is str:
                return dict(zip(l,map(string_match,s)))
            else:
                return dict(zip(l,s))

def _parse_font_type_size(f,s,l,description):
    # Font parsing for openGL
    def string_match(f,s):
        if f =='Helvetica':
            if s ==10:
                return GLUT.GLUT_BITMAP_HELVETICA_10
            elif s ==12:
                return GLUT.GLUT_BITMAP_HELVETICA_12
            elif s == 18:
                return GLUT.GLUT_BITMAP_HELVETICA_18
            else:
                raise RuntimeError("Cannot parse font " + f + ", size " + str(s))
        elif f =='Times':
            if s ==10:
                return GLUT.GLUT_BITMAP_TIMES_ROMAN_10
            elif s == 24:
                return GLUT.GLUT_BITMAP_TIMES_ROMAN_24
            else:
                raise RuntimeError("Cannot parse font " + f + ", size " + str(s))
        else:
            raise RuntimeError("Cannot parse font type " + f)
    if type(f) is str:
        return defaultdict(repeat(string_match(f,s)).next)
    elif type(f) is dict:
        d = {}
        for x in f:
            d[x] = string_match(f[x],s)
        return d
    else:
        if len(f) != len(s) != n:
            raise RuntimeError("Fonts and font size must match " + description + "Dimention")
        else:
            return dict(zip(l,map(string_match,f,s)))

def _calc_self_loop_ctrl_points(G,pos,edgelist,r):
    d = {}
    times_seen = {}
    r_e = {}
    for e in edgelist:
        if (e[0],e[1]) in times_seen:
            times_seen[(e[0],e[1])] += 1
        else:
            times_seen[(e[0],e[1])] = 0
            r_e[(e[0],e[1])] = r

        new_r = r_e[(e[0],e[1])]
        if len(pos[e[0]])==3:
            (x,y,z) = pos[e[0]]
            if times_seen[(e[0],e[1])] % 4==0:
                d[e] = [pos[e[0]],(x-new_r,y+new_r,z),(x+new_r,y+new_r,z),pos[e[0]]]
            elif times_seen[(e[0],e[1])] % 4 == 1:
                d[e] = [pos[e[0]],(x-new_r,y-new_r,z),(x+new_r,y-new_r,z),pos[e[0]]]
            elif times_seen[(e[0],e[1])] % 4 == 2:
                d[e] = [pos[e[0]],(x-new_r,y+new_r,z),(x-new_r,y-new_r,z),pos[e[0]]]
            else:
                d[e] = [pos[e[0]],(x+new_r,y+new_r,z),(x+new_r,y-new_r,z),pos[e[0]]]
                r_e[(e[0],e[1])] += .5*r
        else:
            (x,y) = pos[e[0]]
            if times_seen[(e[0],e[1])] % 4 == 0:
                d[e] = [(x,y,0),(x-new_r,y+new_r,0),(x+new_r,y+new_r,0),(x,y,0)]
            elif times_seen[(e[0],e[1])] % 4 == 1:
                d[e] = [(x,y,0),(x-new_r,y-new_r,0),(x+new_r,y-new_r,0),(x,y,0)]
            elif times_seen[(e[0],e[1])] % 4 == 2:
                d[e] = [(x,y,0),(x-new_r,y+new_r,0),(x-new_r,y-new_r,0),(x,y,0)]
            else:
                d[e] = [(x,y,0),(x+new_r,y+new_r,0),(x+new_r,y-new_r,0),(x,y,0)]
                r_e[(e[0],e[1])] += .2*r
    return d

def _calc_multi_edge_ctrl_points(G,pos,edgelist,r):
    from math import sqrt
    d = {}
    times_seen = {}
    r_e = {}
    norm_v = {}
    mid = {}
    base_points = {}
    for e in edgelist:
        if e[0]==e[1]:
            continue
        e_p = (e[0],e[1])
        fs_ep = frozenset(e_p)
        if fs_ep in times_seen:
            times_seen[fs_ep] += 1
        else:
            times_seen[fs_ep] = 1
            r_e[fs_ep] = r
        if times_seen[fs_ep] == 2:
            norm_v[fs_ep] = []
            mid[fs_ep] = []
            base_points[fs_ep] = []
            for x in range(len(pos[e[0]])):
                norm_v[fs_ep].append(pos[e[1]][x]-pos[e[0]][x])
                mid[fs_ep].append((pos[e[1]][x] + pos[e[0]][x])/2.)
            if len(pos[e[0]]) == 3:
                try:
                    a = 0
                    c = 1./sqrt((norm_v[fs_ep][2]/norm_v[fs_ep][1])**2 + 1)
                    b = -c*(norm_v[fs_ep][2]/norm_v[fs_ep][1])
                    base_points[fs_ep].append((a,b,c))
                    base_points[fs_ep].append((a,-b,-c))
                except:
                    pass
                try:
                    b = 0
                    c = 1./sqrt((norm_v[fs_ep][2]/norm_v[fs_ep][0])**2 + 1)
                    a = -c*(norm_v[fs_ep][2]/norm_v[fs_ep][0])
                    base_points[fs_ep].append((a,b,c))
                    base_points[fs_ep].append((-a,b,-c))
                except:
                    pass
                try:
                    c = 0
                    b = 1./sqrt((norm_v[fs_ep][1]/norm_v[fs_ep][0])**2 + 1)
                    a = -b*(norm_v[fs_ep][1]/norm_v[fs_ep][0])
                    base_points[fs_ep].append((a,b,c))
                    base_points[fs_ep].append((-a,-b,c))
                except:
                   pass
            if len(pos[e[0]])==2:
                mid[fs_ep].append(0)
                try:
                    b = 1./sqrt((norm_v[fs_ep][1]/norm_v[fs_ep][0])**2 + 1)
                    a = -b*((norm_v[fs_ep][1]/norm_v[fs_ep][0]))
                    base_points[fs_ep].append((a,b,0))
                    base_points[fs_ep].append((-a,-b,0))
                except:
                    base_points[fs_ep].append((0,1,0))
                    base_points[fs_ep].append((0,-1,0))
                    
        if times_seen[fs_ep] >= 2:
            bp_index = (times_seen[fs_ep]-2) % len(base_points[fs_ep])
            if len(pos[e[0]]) == 2:
                pos0 = tuple(list(pos[e[0]])+ [0])
                pos1 = tuple(list(pos[e[1]])+ [0])
            else:
                pos0 = pos[e[0]]
                pos1 = pos[e[1]]
            d[e] = [pos0,
                    tuple(map(lambda (m,x): m + x*r_e[fs_ep],
                              zip(mid[fs_ep],base_points[fs_ep][bp_index]))),
                    pos1]
            if bp_index == len(base_points[fs_ep]) -1:
                r_e[fs_ep] += .2*r
            
    return d

        
def _calc_edge_labels(G,edgelist):
    d = {}
    for e in edgelist:
        if G.is_multigraph():
            if len(G.edge[e[0]][e[1]]) == 1:
                d[e] = str((e[0],e[1]))
            else:
                d[e] = str(e)
        else:
            d[e] = str(e)
    return d

def _calc_edge_label_pos(G,pos,edgelist,r,multi_edge_dict):
    d = {}
    times_seen = {}
    r_e = {}
    for e in edgelist:
        if (e[0],e[1]) in times_seen:
            times_seen[(e[0],e[1])] += 1
        else:
            times_seen[(e[0],e[1])] = 0
            r_e[(e[0],e[1])] = r

        new_r = r_e[(e[0],e[1])]
        
        if e[0]==e[1]:
            position = list(pos[e[0]])
            if times_seen[(e[0],e[1])] % 4 == 0:
                position[1] += .75*new_r #from the Bernstein polynomial
            elif times_seen[(e[0],e[1])] % 4 == 1:
                position[1] -= .75*new_r
            elif times_seen[(e[0],e[1])] % 4 == 2:
                position[0] -= .75*new_r
            else:
                position[0] += .75*new_r
                r_e[(e[0],e[1])] += .2*r
        elif e in multi_edge_dict:
            position = [.25*p0+.5*pc+.25*p1 for (p0,pc,p1) in zip(*multi_edge_dict[e])]
        else:
            position = []
            for x in range(len(pos[e[0]])):
                position.append((pos[e[0]][x]+pos[e[1]][x])/2.)
        d[e] = position
    return d
            
            
def _calc_arrow_points(pos,
                       edgelist,
                       arrow_size_dict,
                       self_loop_edge_dict,
                       multi_edge_dict,
                       width,
                       height):
    #Pre calculates the arrow positions. Since we don't want them
    #rendered inside the ndoes, we have to back them off by an amount
    #equal to the node size. OpenGL also by default draws everything
    # at 0,0,0, so we have to find out how much we have to rotate and
    # translate the axis to put them in the correct place. This requires
    # some linear algebra, but happily we only ahve to do it once.
    from math import sqrt

    def dot(p1,p2):
        s = 0
        for k in range(len(p2)):
            s += p1[k]*p2[k]
        return s
    def normalize(p):
        v_len = sqrt(dot(p,p))
        if v_len == 0:
            return tuple([0 for _ in range(len(p)-1)]+[-1])
        return tuple(map(lambda x: x/v_len, p))

    def cross(a,b):
        return (a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0])

    d = {}
    c = sqrt(width**2 + height**2 + 1)/(2.0)
    d['__c'] = c
    for e in edgelist:
        if e in self_loop_edge_dict:
            xi = .1*self_loop_edge_dict[e][1][0] + .9*self_loop_edge_dict[e][2][0]
            yi = .1*self_loop_edge_dict[e][1][1] + .9*self_loop_edge_dict[e][2][1]
            zi = .1*self_loop_edge_dict[e][1][2] + .9*self_loop_edge_dict[e][2][2]
        elif e in multi_edge_dict:
            (xi,yi,zi) = multi_edge_dict[e][1]
        else:
            (xi,yi,zi) = pos[e[0]]
            
        (xj,yj,zj) = pos[e[1]]
        try:
            n = (1,-(xj-xi)/(yj-yi),0)
        except:
            try:
                n = (1,0,-(xj-xi)/(zj-zi))
            except:
                d[e] = ([[(0,1,0),(xj,yj,zj),(0,0,1)],
                         [(0,0,1),(xj,yj,zj),(0,-1,0)],
                         [(0,-1,0),(xj,yj,zj),(0,0,-1)],
                         [(0,0,-1),(xj,yj,zj),(0,1,0)]],(1,0,0))
                continue
        v = normalize((xi-xj,yi-yj,zi-zj))

        p1 = normalize(n)
        p2 = tuple(-x for x in p1)
        p3 = normalize(cross(p1,v))
        p4 = tuple(-x for x in p3)

        d[e] = ([[p1,(xj,yj,zj),p3],
                 [p3,(xj,yj,zj),p2],
                 [p2,(xj,yj,zj),p4],
                 [p4,(xj,yj,zj),p1]],v)
    return d

def _graph_render_func(window):
    # Does the rendering grunt work, with the actual parsed values from
    # the calling function

    # We want to depth test between groups of objects being drawn
    GL.glEnable(GL.GL_DEPTH_TEST)
    
    # These should be at the back
    _draw_edges(window)

    # Then nodes with proper depth fixing
    if window.data['with_node_labels']:
        _draw_node_labels(window)
    _draw_nodes(window)
    if window.data['with_arrows']:
        _draw_arrows(window)
    #On top of everything so far


    #On top of those
    if window.data['with_edge_labels']:
        _draw_edge_labels(window)



    GL.glDisable(GL.GL_DEPTH_TEST)
    # Draw the arrows on top of the edges
    
    #Draw the nodes on top of those
    
    #labels on top of everything

def _draw_edges(window):
    GL.glEnable(GL.GL_LINE_STIPPLE)
    GL.glDisable(GL.GL_DEPTH_TEST)
    for e in window.data['edgelist']:
        GL.glLineWidth(window.data['edge_thickness'][e])
        GL.glLineStipple(*window.data['edge_style'][e])
        GL.glColor(*window.data['edge_color'][e])
        if e[0]==e[1]:
            GL.glMap1f(GL.GL_MAP1_VERTEX_3,0.0,1.0,window.data['self_loop_ctrl_points'][e])
            GL.glEnable(GL.GL_MAP1_VERTEX_3)
            GL.glBegin(GL.GL_LINE_STRIP)
            GL.glColor(*window.data['edge_color'][e])
            for k in [x*(1./50) for x in range(50+1)]:
                GL.glEvalCoord1f(k)
            GL.glEnd()
        elif 'multi_edge_ctrl_points' in window.data and e in window.data['multi_edge_ctrl_points']:
            GL.glMap1f(GL.GL_MAP1_VERTEX_3,0.0,1.0,window.data['multi_edge_ctrl_points'][e])
            GL.glEnable(GL.GL_MAP1_VERTEX_3)
            GL.glBegin(GL.GL_LINE_STRIP)
            GL.glColor(*window.data['edge_color'][e])
            for k in [x*(1./50) for x in range(50+1)]:
                GL.glEvalCoord1f(k)
            GL.glEnd()
        else:
            GL.glBegin(GL.GL_LINES)
            GL.glColor(*tuple(window.data['edge_color'][e]))
            GL.glVertex(*tuple(window.data['pos'][e[0]]))
            GL.glVertex(*tuple(window.data['pos'][e[1]]))
            GL.glEnd()
    GL.glEnable(GL.GL_DEPTH_TEST)

def _draw_arrows(window):
    def scale(p,alpha):
        return tuple(x*alpha for x in p)
    def translate(a,b):
        return tuple(x1+x2 for (x1,x2) in zip(a,b))
    c = window.data['arrow_points']['__c']*window._scale
    #GL.glDisable(GL.GL_DEPTH_TEST)

    for e in window.data['edgelist']:
        GL.glBegin(GL.GL_TRIANGLES)
        s1 = window.data['arrow_size'][e]/c
        s2 = window.data['arrow_size'][e]/(3.*c)
        #I have no idea why 1.5 is correct here but it seemst to be.
        r = (window.data['node_size'][e[1]]/1.5)+(window.data['node_border_size'][e[1]]/1.5)
        GL.glColor(*window.data['edge_color'][e])
        v = window.data['arrow_points'][e][1]
        for [p1,p2,p3] in window.data['arrow_points'][e][0]:
            p2 = translate(p2,scale(v,r/c))
            
            p1 = scale(p1,s2)
            p1 = translate(p1,p2)
            p1 = translate(p1,scale(v,s1))

            p3 = scale(p3,s2)
            p3 = translate(p3,p2)
            p3 = translate(p3,scale(v,s1))
            GL.glVertex3d(*p1)
            GL.glVertex3d(*p2)
            GL.glVertex3d(*p3)
        GL.glEnd()
    #GL.glEnable(GL.GL_DEPTH_TEST)
            
    
    """
    #GL.glDisable(GL.GL_DEPTH_TEST)
    for (i,j) in edgelist:
        d = []
        for r in range(len(pos[i])):
            d.append(pos[i][r]-pos[j][r])
        d = tuple(d + [0])
        GL.glPushMatrix()

        GL.glTranslate(arrow_points[k][0][0],
                       arrow_points[k][0][1],
                       arrow_points[k][0][2])
        GL.glScale(scale,scale,scale)
        GL.glRotate(arrow_points[k][1],-d[1],d[0],0)

        GL.glColor(*edge_color[k])
        vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
        arrow_size_scale = 2./(vp[2]+vp[3])
        if fancy:
            cone = GLU.gluNewQuadric()
            GLU.gluQuadricNormals(cone,GLU.GLU_SMOOTH)
            GLU.gluQuadricTexture(cone,GLU.GLU_TRUE)
            GLU.gluCylinder(cone,
                            0,
                            arrow_size_scale*arrow_size[k]/3.,
                            arrow_size_scale*arrow_size[k],
                            32,
                            32)
        else:
            s1 = arrow_size[k]*arrow_size_scale
            s2 = arrow_size_scale*arrow_size[k]/3.
            GL.glBegin(GL.GL_POLYGON);
            GL.glVertex3d(0, 0, 0);
            GL.glVertex3d(-s2, s2, s1);
            GL.glVertex3d(-s2, -s2, s1);

            GL.glVertex3d(0, 0, 0);
            GL.glVertex3d(-s2, s2, s1);
            GL.glVertex3d(s2, s2, s1);

            GL.glVertex3d(0, 0, 0);
            GL.glVertex3d(s2, s2, s1);
            GL.glVertex3d(s2, -s2, s1);

            GL.glVertex3d(0, 0, 0);
            GL.glVertex3d(s2, -s2, s1);
            GL.glVertex3d(-s2, -s2, s1);

            GL.glVertex3d(-s2, -s2, s1);
            GL.glVertex3d(s2, -s2, s1);
            GL.glVertex3d(s2, s2, s1);
            GL.glVertex3d(-s2, s2, s1);
            GL.glEnd();
        GL.glPopMatrix()
        k+=1
    """
    #GL.glEnable(GL.GL_DEPTH_TEST)


def _draw_node_labels(window):
    for n in window.data['nodelist']:
        GL.glColor(*window.data['node_label_colors'][n])
        GL.glRasterPos(*window.data['pos'][n])
        GLUT.glutBitmapString(window.data['node_label_font'][n],
                              window.data['node_labels'][n])
    
def _draw_edge_labels(window):
    GL.glDisable(GL.GL_DEPTH_TEST)
    for e in window.data['edgelist']:
        GL.glRasterPos(*window.data['edge_label_pos'][e])
        GL.glColor(*window.data['edge_label_colors'][e])
        GLUT.glutBitmapString(window.data['edge_label_font'][e],
                              window.data['edge_labels'][e])
    GL.glEnable(GL.GL_DEPTH_TEST)

        
def _draw_nodes(window):
    #GL.glDisable(GL.GL_DEPTH_TEST)

    for n in window.data['nodelist']:
        #Ensures we blend together between node border and node properly
        #If we leave it as is, the background color bleeds between the two
        # and it looks bad. This probably slow...
        #GL.glBlendFunc(GL.GL_ONE,GL.GL_CONSTANT_COLOR)
        #GL.glBlendColor(*window.data['node_color'][n])
	GL.glPointSize(window.data['node_size'][n])
        GL.glBegin(GL.GL_POINTS)
        GL.glColor(*window.data['node_color'][n])
        GL.glVertex(*tuple(window.data['pos'][n]))
        GL.glEnd()
        #Back to what we had
        #GL.glBlendFunc(GL.GL_ONE,GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glPointSize(window.data['node_size'][n]+window.data['node_border_size'][n])
        GL.glBegin(GL.GL_POINTS)
        GL.glColor(*window.data['node_border_color'][n])
        GL.glVertex(*tuple(window.data['pos'][n]))
        GL.glEnd()
    GL.glBlendColor(0.,0.,0.,0.)
    
    #GL.glEnable(GL.GL_DEPTH_TEST)

def _comm_thread(window):
    window.ctx = zmq.Context()
    window.sckt = window.ctx.socket(zmq.PAIR)
    window.sckt.connect('tcp://127.0.0.1:%s'%window.data['_port'])
    while True:
        prop = window.sckt.recv_pyobj()
        if prop =="exit":
            window._on_exit()
            break
        k = window.sckt.recv_pyobj()
        v = window.sckt.recv_pyobj()
        try:
            window.data[prop][k] = v
        except:
            pass
        
    
if __name__ == "__main__":
    # A few examples
    G = nx.barabasi_albert_graph(10,2)
    draw_opengl(G)
    G = nx.DiGraph(G)
    draw_opengl(G)
    G = nx.fast_gnp_random_graph(100,1/100.)
    ns = [(G.degree(n)+1)**2 for n in G.nodes()]
    draw_opengl(G,with_node_labels=False,node_size=ns)
    
