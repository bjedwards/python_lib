import networkx as nx
import matplotlib.colors as colors
import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

import gl_windows as glw

def draw_opengl_3d(G,
                   pos=None,
                   nodelist=None,
                   node_size=20,
                   node_color=(1.,0.,0.,1.),
                   node_border_size=3,
                   edgelist=None,
                   edge_color=(0.,0.,0.,1.),
                   edge_style='solid',
                   edge_thickness=1.0,
                   with_arrows=True,
                   with_node_labels=True,
                   node_labels=None,
                   node_font_size=12,
                   node_label_color=(0.,0.,0.,1.),
                   with_edge_labels=False,
                   edge_labels=None,
                   edge_font_size=12,
                   edge_font_color=(0.,0.,0.,1.)):
    try:
        import OpenGL.GL as GL
        import OpenGL.GLU as GLU
        import OpenGL.GLUT as GLUT
    except ImportError:
        raise ImportError("PyOpenGl, with working OpenGL, GLU, and GLUT libraries")
    if pos is None:
        pos=nx.drawing.spring_layout(G,dim=3)
        for n in pos:
            pos[n] = map(lambda x: x-.5,pos[n])
    if nodelist is None:
        nodelist = G.nodes()

    node_size = parse_numeric_value(node_size,len(nodelist),"Node Size")
    node_color = parse_color_value(node_color,len(nodelist),"Node Color")
    node_border_size = parse_numeric_value(node_border_size,
                                           len(nodelist),
                                           "Node Border Size")
    if edgelist is None:
        edgelist = G.edges()
    edge_color = parse_color_value(edge_color,len(edgelist),"Edge Color")
    edge_thickness = parse_numeric_value(edge_thickness,
                                         len(edgelist),
                                         "Edge Thickness")
    if with_node_labels and node_labels is None:
        node_labels = map(str,nodelist)

        node_label_color = parse_color_value(node_label_color,
                                             len(node_labels),
                                             "Node Label Color")
    glw.gl_3d_window(background_color=(1.,1.,1.,1.),
                     refresh=10,
                     height=480,
                     width=480,
                     render_func=rend_func,
                     rf_args=(pos,nodelist,node_size,node_color,node_border_size,edgelist,edge_color,edge_style,edge_thickness,with_arrows,with_node_labels,node_labels,node_font_size,node_label_color, with_edge_labels,edge_labels,edge_font_size,edge_font_color))


def draw_opengl_2d(G,
                   pos=None,
                   nodelist=None,
                   node_size=20,
                   node_color=(1.,0.,0.,1.),
                   node_border_size=3,
                   edgelist=None,
                   edge_color=(0.,0.,0.,1.),
                   edge_style='solid',
                   edge_thickness=1.0,
                   with_arrows=True,
                   with_node_labels=True,
                   node_labels=None,
                   node_font_size=12,
                   node_label_color=(0.,0.,0.,1.),
                   with_edge_labels=False,
                   edge_labels=None,
                   edge_font_size=12,
                   edge_font_color=(0.,0.,0.,1.)):
    try:
        import OpenGL.GL as GL
        import OpenGL.GLU as GLU
        import OpenGL.GLUT as GLUT
    except ImportError:
        raise ImportError("PyOpenGl, with working OpenGL, GLU, and GLUT libraries")
    if pos is None:
        pos=nx.drawing.spring_layout(G,dim=2)
        for n in pos:
            pos[n] = map(lambda x: x-.5,pos[n])
    if nodelist is None:
        nodelist = G.nodes()

    node_size = parse_numeric_value(node_size,len(nodelist),"Node Size")
    node_color = parse_color_value(node_color,len(nodelist),"Node Color")
    node_border_size = parse_numeric_value(node_border_size,
                                           len(nodelist),
                                           "Node Border Size")
    if edgelist is None:
        edgelist = G.edges()
    edge_color = parse_color_value(edge_color,len(edgelist),"Edge Color")
    edge_thickness = parse_numeric_value(edge_thickness,
                                         len(edgelist),
                                         "Edge Thickness")
    if with_node_labels and node_labels is None:
        node_labels = map(str,nodelist)

        node_label_color = parse_color_value(node_label_color,
                                             len(node_labels),
                                             "Node Label Color")
    glw.gl_2d_window(background_color=(1.,1.,1.,1.),
                     refresh=10,
                     height=480,
                     width=480,
                     render_func=rend_func,
                     rf_args=(pos,nodelist,node_size,node_color,node_border_size,edgelist,edge_color,edge_style,edge_thickness,with_arrows,with_node_labels,node_labels,node_font_size,node_label_color, with_edge_labels,edge_labels,edge_font_size,edge_font_color))

def parse_numeric_value(x,n,description):
    if type(x) is int or type(x) is float:
        return [x]*n
    else:
        if not len(x)==n:
            raise RuntimeError("Dimensions of " + description + " do not match")

def parse_color_value(c,n,description):
    if type(c) is tuple:
        return [c]*n
    elif type(c) is string:
        return [colors.colorConverter.to_rgba(c)]*n
    else:
        if not len(c)==n:
            raise RuntimeError("Dimensions of " + description + " do not match")

def rend_func(window,
              pos,
              nodelist,
              node_size,
              node_color,
              node_border_size,
              edgelist,
              edge_color,
              edge_style,
              edge_thickness,
              with_arrows,
              with_node_labels,
              node_labels,
              node_font_size,
              node_label_color,
              with_edge_labels,
              edge_labels,
              edge_font_size,
              edge_font_color):
    
    GL.glEnable(GL.GL_POINT_SMOOTH)
    GL.glEnable(GL.GL_LINE_SMOOTH)
    GL.glEnable(GL.GL_DEPTH_TEST)
    draw_edges(pos,edgelist,edge_color,edge_thickness)
    """
    if with_arrows:
        draw_arrows(pos,
                    edgelist,
                    edge_color,
                    1/window.scale,
                    window.lights,
                    p_size)
                    """
    draw_nodes(pos,nodelist,node_color,node_size,node_border_size)
    if with_node_labels:
        draw_node_labels(pos,nodelist,node_labels,node_label_color)

def draw_edges(pos,edgelist,edge_colors,edge_thickness):
    #Parse color stuff later, for single colors etc
    k = 0
    for (i,j) in edgelist:
        GL.glLineWidth(edge_thickness[k])
        GL.glBegin(GL.GL_LINES)
        GL.glColor(*tuple(edge_colors[k]))
        GL.glVertex(*tuple(pos[i]))
        GL.glVertex(*tuple(pos[j]))
        GL.glEnd()
        k+=1
"""
def draw_arrows(pos,
                edgelist,
                edge_color,
                scale,
                lights,
                p_size):
    GL.glDisable(GL.GL_DEPTH_TEST)
    if lights:
        GL.glEnable(GL.GL_LIGHTING)
    for (i,j) in edges:
        d = pos[i][0:3] - pos[j][0:3]
        c = np.array([0,0,1])
        mm = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        pm = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
        pjwx,pjwy,pjwz = GLU.gluProject(pos[j][0],pos[j][1],pos[j][2],mm,pm,vp)
        piwx,piwy,piwz = GLU.gluProject(pos[i][0],pos[i][1],pos[i][2],mm,pm,vp)
        nw = np.array([pjwx-piwx,pjwy-piwy,pjwz,piwz])
        nw = nw/np.sqrt(np.dot(nw,nw.conj()))
        cone_stopw = [pjwx-(p_size/2.)*nw[0],
                      pjwy-(p_size/2.)*nw[1],
                      pjwz-(p_size/2.)*nw[2]]
        cone_stop = np.array([0.,0.,0.])
        cone_stop[0],cone_stop[1],cone_stop[2] = GLU.gluUnProject(cone_stopw[0],
                                                                  cone_stopw[1],
                                                                  cone_stopw[2],
                                                                  mm,
                                                                  pm,
                                                                  vp)
        
        GL.glPushMatrix()
        theta = np.arccos(np.dot(d,c))*(180./np.pi)
        GL.glTranslate(cone_stop[0],cone_stop[1],cone_stop[2])
        GL.glRotate(theta,pos[j][1]-pos[i][1],pos[i][0]-pos[j][0],0)
        GL.glScale(scale,scale,scale)
        cone = GLU.gluNewQuadric()
        GLU.gluQuadricNormals(cone,GLU.GLU_SMOOTH)
        GLU.gluQuadricTexture(cone,GLU.GLU_TRUE)
        GLU.gluCylinder(GLU.gluNewQuadric(),0,np.sqrt(p_size/20.)*.025,np.sqrt(p_size/20.)*.05,32,32)
        GL.glPopMatrix()
    if lights:
        GL.glDisable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_DEPTH_TEST)
"""
def draw_node_labels(pos,nodelist,node_labels,node_label_colors):
    GL.glDisable(GL.GL_DEPTH_TEST)
    i = 0
    for n in nodelist:
        GL.glRasterPos(*tuple(pos[n]))
        GL.glColor(*node_label_colors[i])
        GLUT.glutBitmapString(GLUT.GLUT_BITMAP_HELVETICA_10,node_labels[i])
        i+=1
    GL.glEnable(GL.GL_DEPTH_TEST)
        
def draw_nodes(pos,nodelist,node_color,node_size,node_border_size):
    GL.glDisable(GL.GL_DEPTH_TEST)
    i = 0
    for n in nodelist:
        GL.glPointSize(node_size[i]+node_border_size[i])
        GL.glBegin(GL.GL_POINTS)
        GL.glColor(0.,0.,0.)
        GL.glVertex(*tuple(pos[n]))
        GL.glEnd()

        GL.glPointSize(node_size[i])
        GL.glBegin(GL.GL_POINTS)
        GL.glColor(*node_color[i])
        GL.glVertex(*tuple(pos[n]))
        GL.glEnd()
        i+=1
    GL.glEnable(GL.GL_DEPTH_TEST)


if __name__ == "__main__":
    try:
        num = int(sys.argv[1])
    except:
        num = 500
    try:
        refresh = int(sys.argv[2])
    except:
        refresh = 15
    try:
        p_size = int(sys.argv[3])
    except:
        p_size = 20
    try:
        with_labels = sys.argv[4] == "True"
    except:
        with_labels = True
    try:
        with_arrows = sys.argv[5] == "True"
    except:
        with_arrows = False
    #pos = np.ndarray((num,4),dtype=np.float32)
    #pos[:,0] = np.random.random_sample((num,)) - 0.5
    #pos[:,1] = np.random.random_sample((num,)) - 0.5
    #pos[:,2] = np.random.random_sample((num,)) - 0.5
    #pos[:,3] = 1.0
    
    #col[:,0] = 1.0
    #col[:,1] = 0.0
    #col[:,2] = 0.0
    #col[:.3] = 0.74
    #G = nx.barabasi_albert_graph(num,2)
    G = nx.fast_gnp_random_graph(num,5./num)
    pos_dict = nx.fruchterman_reingold_layout(G,dim=3)
    #pos_dict = nx.spectral_layout(G,dim=3)
    sort_nodes = sorted(G.nodes())
    xmid = np.mean([pos_dict[n][0] for n in sort_nodes])
    ymid = np.mean([pos_dict[n][1] for n in sort_nodes])
    zmid = np.mean([pos_dict[n][2] for n in sort_nodes])
    pos = np.array([pos_dict[n] - np.array([xmid,ymid,zmid]) for n in sort_nodes])
    col = np.ndarray((num,4),dtype=np.float32)
    dc = nx.degree_centrality(G)
    max_dc = np.max(dc.values())
    deg_col = np.array([dc[n]/max_dc for n in sort_nodes])
    #rand_col = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)/np.sqrt(3*(.5**2))
    col[:,0] = np.min([4*deg_col - 1.5,-4 * deg_col + 4.5],axis=0)
    col[:,1] = np.min([4*deg_col - 0.5,-4 * deg_col + 3.5],axis=0)
    col[:,2] = np.min([4*deg_col + 0.5,-4 * deg_col + 2.5],axis=0)
    gl_3d_window(render_func = rend_func,rf_args=(pos,col,p_size,G.edges(),with_labels,with_arrows),refresh=refresh)
