def rend_func(window,pos,col,p_size,edges,with_labels=False,with_arrows=True,scale=1.0):
    GL.glEnable(GL.GL_POINT_SMOOTH)
    GL.glPointSize(p_size)
    GL.glEnable(GL.GL_DEPTH_TEST)
    draw_edges(pos,edges)
    if with_arrows:
        draw_arrows(pos,edges,1/window.scale,window.lights,p_size)
    draw_nodes(pos,col,p_size)
    if with_labels:
        draw_labels(pos,map(str,range(len(pos))))

def draw_labels(pos,labels):
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glColor3f(0.,0.,0.)
    i = 0
    for l in labels:
        GL.glRasterPos3f(pos[i][0],pos[i][1],pos[i][2])
        GLUT.glutBitmapString(GLUT.GLUT_BITMAP_HELVETICA_10,l)
        i+=1
    GL.glEnable(GL.GL_DEPTH_TEST)
        
def draw_nodes(pos,col,p_size):
    GL.glColor3f(1.,1.,1.)
    GL.glPointSize(p_size+3)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glBegin(GL.GL_POINTS)
    for p in pos:
        GL.glVertex3f(p[0],p[1],p[2])
    GL.glEnd()
    i = 0
    GL.glPointSize(p_size)
    GL.glBegin(GL.GL_POINTS)
    for p in pos:
        GL.glColor3f(col[i][0],col[i][1],col[i][2])
        GL.glVertex3f(p[0],p[1],p[2])
        i+=1
    GL.glEnd()
    GL.glEnable(GL.GL_DEPTH_TEST)

def draw_edges(pos,edges,edge_colors=[]):
    #Parse color stuff later, for single colors etc
    if len(edge_colors) < len(edges):
        edge_colors = edge_colors + [np.array([1.,1.,1.])]*(len(edges) - len(edge_colors))
    GL.glBegin(GL.GL_LINES)
    k = 0
    for (i,j) in edges:
        GL.glColor3f(edge_colors[k][0],edge_colors[k][1],edge_colors[k][2])
        GL.glVertex3f(pos[i][0],pos[i][1],pos[i][2])
        GL.glVertex3f(pos[j][0],pos[j][1],pos[j][2])
        k+=1
    GL.glEnd()

def draw_arrows(pos,edges,scale,lights,p_size,edge_colors=[]):
    if len(edge_colors) < len(edges):
        edge_colors = edge_colors + [np.array([1.,1.,1.])]*(len(edges) - len(edge_colors))
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
