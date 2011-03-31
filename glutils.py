import OpenGL.GLUT as GLUT
import OpenGL.GL as GL
import OpenGL.GLU as GLU
from PIL import Image

import os

class gl_3d_window(object):
    def __init__(self,
                 width=1080,
                 height=1080,
                 title="",
                 refresh=15,
                 save_file='./',
                 background_color=(0.,0.,0.,0.),
                 save_file_type=None,
                 render_func=lambda window: None,
                 rf_args=(), rf_kwargs = {}):
        
        pid = os.fork() # This is a hack to make it work in the interpreter
        if not pid==0:  # Mostly because OpenGL doesn't exit nicely
            return

        self.mouse_down = False
        self.mouse_old = [0., 0.]
        self.rotate =[0., 0., 0.]
        self.translate = [0., 0., 0.]
        self.scale = 1.0
        split_path_ext = os.path.splitext(save_file)
        self.save_file = split_path_ext[0]
        if save_file_type is None:
            self.save_file_ext = split_path_ext[1]
            if self.save_file_ext == '':
                self.save_file_ext = '.pdf'
        else:
            self.save_file_ext = save_file_type
        self.save_count = 0
        
        self.width = width
        self.height = height
        self.lights = True
        
        GLUT.glutInit()
        
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(self.width, self.height)
        GLUT.glutInitWindowPosition(0, 0)
        self.win = GLUT.glutCreateWindow(title)

        #gets called by GLUT every frame
        GLUT.glutDisplayFunc(self.draw(render_func,*rf_args,**rf_kwargs))

        #handle user input
        GLUT.glutKeyboardFunc(self.on_key)
        GLUT.glutMouseFunc(self.on_click)
        GLUT.glutMotionFunc(self.on_mouse_motion)
        GLUT.glutCloseFunc(self.on_exit)
        
        #this will call draw every refresh ms
        GLUT.glutTimerFunc(refresh, self.timer, refresh)

        #setup OpenGL scene
        self.glinit()

        GL.glClearColor(*background_color)
        #set up initial conditions
        #create our OpenCL instance
        GLUT.glutMainLoop()
        

    def glinit(self):
        GL.glViewport(0, 0, self.width, self.height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1.,1.,-1,1.,-1000.,1000.)
        GL.glMatrixMode(GL.GL_MODELVIEW)


    ###GL CALLBACKS
    def timer(self, t):
        GLUT.glutTimerFunc(t, self.timer, t)
        GLUT.glutPostRedisplay()

    def on_exit(x):
        os._exit(0)

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            os._exit(0)
        elif args[0] == 'r':
            self.rotate = [0., 0., 0.]
            self.translate = [0., 0., 0.]
            self.scale = 1.0
        elif args[0] == 'l':
            if self.lights:
                self.lights = False
            else:
                self.lights = True
        elif args[0] == 's':
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            pixel_array = GL.glReadPixels(0,0,vp[2],vp[3],GL.GL_RGB,GL.GL_UNSIGNED_BYTE)

            pilImage = Image.fromstring(mode="RGB",size=(vp[3],vp[2]),data=pixel_array)
            pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
            pilImage.save(self.save_file + str(self.save_count) + '.png')
            self.save_count += 1 
            

    def on_click(self, button, state, x, y):
        if state == GLUT.GLUT_DOWN:
            self.mouse_down = True
            self.button = button
            if self.button == 3 or self.button == 5:
                self.scale *= 1.1
            if self.button == 4 or self.button == 6:
                self.scale *= 0.9
        else:
            self.mouse_down = False
        self.mouse_old[0] = x
        self.mouse_old[1] = y

    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old[0]
        dy = y - self.mouse_old[1]
        if self.mouse_down and self.button == 0: #left button
            self.rotate[0] += dy * .1
            self.rotate[1] += dx * .1
        elif self.mouse_down and self.button == 2: #right button
            self.translate[0] += dx * .001
            self.translate[1] -= dy * .001
        self.mouse_old[0] = x
        self.mouse_old[1] = y

    ###END GL CALLBACKS

    def draw(self,f,*args,**kwargs):
        """Render the particles"""        
        #update or particle positions by calling the OpenCL kernel
        def draw_func():
            GL.glFlush()

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

            #handle mouse transformations
            GL.glTranslatef(0.,0.,-2)
            GL.glRotatef(self.rotate[0], 1, 0, 0)
            GL.glRotatef(self.rotate[1], 0, 1, 0) #we switched around the axis so make this rotate_z
            GL.glTranslatef(self.translate[0], self.translate[1], self.translate[2])
            GL.glScale(self.scale,self.scale,self.scale)
            GL.glShadeModel(GL.GL_SMOOTH)
            GL.glLightfv(GL.GL_LIGHT0,GL.GL_POSITION, (1.,1.,0.))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.,0.,0.,1.))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (1.0,1.0,1.0,1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR,(1.,1.,1.,1.))
            GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT,(.2,.2,.2,1.0))
            GL.glEnable(GL.GL_LIGHT0)
            GL.glEnable(GL.GL_COLOR_MATERIAL)

            f(self,*args,**kwargs)

            GLUT.glutSwapBuffers()
        return draw_func

class gl_2d_window(object):
    def __init__(self,
                 width=1080,
                 height=1080,
                 title="",
                 refresh=15,
                 save_file='./',
                 
                 save_file_type=None,
                 background_color = (0.,0.,0.,0.),
                 render_func=lambda window: None,
                 rf_args=(),
                 rf_kwargs = {}):

        pid = os.fork()
        if not pid==0:
            return None
        self.mouse_down = False
        self.mouse_old = [0., 0.]
        self.rotate = [0., 0., 0.]
        self.translate = [0., 0., 0.]
        self.scale = 1.0

        split_path_ext = os.path.splitext(save_file)
        self.save_file = split_path_ext[0]
        if save_file_type is None:
            self.save_file_ext = split_path_ext[1]
            if self.save_file_ext == '':
                self.save_file_ext = '.png'
        else:
            self.save_file_ext = save_file_type
        self.save_count = 0

        self.width = width
        self.height = height
        self.lights = True
        
         

        GLUT.glutInit()
        
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE)
        GLUT.glutInitWindowSize(self.width, self.height)
        GLUT.glutInitWindowPosition(0, 0)
        self.win = GLUT.glutCreateWindow(title)

        #gets called by GLUT every frame
        GLUT.glutDisplayFunc(self.draw(render_func,*rf_args,**rf_kwargs))

        #handle user input
        GLUT.glutKeyboardFunc(self.on_key)
        GLUT.glutMouseFunc(self.on_click)
        GLUT.glutMotionFunc(self.on_mouse_motion)
        GLUT.glutCloseFunc(self.on_exit)
        
        #this will call draw every refresh ms
        GLUT.glutTimerFunc(refresh, self.timer, refresh)

        #setup OpenGL scene
        self.glinit()
        GL.glClearColor(*background_color)
        #set up initial conditions
        #create our OpenCL instance
        GLUT.glutMainLoop()
        
    def glinit(self):
        GL.glViewport(0, 0, self.width, self.height)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluOrtho2D(-1.,1.,-1.,1.)

    ###GL CALLBACKS
    def timer(self, t):
        GLUT.glutTimerFunc(t, self.timer, t)
        GLUT.glutPostRedisplay()

    def on_exit(x):
        os._exit(0)
        
    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            self.on_exit()
        elif args[0] == 'r':
            self.rotate = [0., 0., 0.]
            self.translate = [0., 0., 0.]
            self.scale = 1.0
        elif args[0] == 'l':
            if self.lights:
                self.lights = False
            else:
                self.lights = True
        elif args[0] == 's':
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            pixel_array = GL.glReadPixels(0,0,vp[2],vp[3],GL.GL_RGB,GL.GL_UNSIGNED_BYTE)

            pilImage = Image.fromstring(mode="RGB",size=(vp[3],vp[2]),data=pixel_array)
            pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
            pilImage.save(self.save_file + str(self.save_count) + self.save_file_ext)
            self.save_count += 1 
            
            

    def on_click(self, button, state, x, y):
        if state == GLUT.GLUT_DOWN:
            self.mouse_down = True
            self.button = button
            if self.button == 3 or self.button == 5:
                self.scale *= 1.1
            if self.button == 4 or self.button == 6:
                self.scale *= 0.9
        else:
            self.mouse_down = False
        self.mouse_old[0] = x
        self.mouse_old[1] = y

    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old[0]
        dy = y - self.mouse_old[1]
        if self.mouse_down and self.button == 0: #left button
            self.rotate[0] -= dy * .2
            self.rotate[1] += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate[0] += dx * .001
            self.translate[1] -= dy * .001
        self.mouse_old[0] = x
        self.mouse_old[1] = y

    ###END GL CALLBACKS


    def draw(self,f,*args,**kwargs):
        """Render the particles"""        
        #update or particle positions by calling the OpenCL kernel
        def draw_func():
            GL.glFlush()

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

            #handle mouse transformations
            GL.glTranslatef(0.,0.,0.)
            GL.glRotatef(self.rotate[0], 0, 0, 1)
            GL.glRotatef(self.rotate[1], 0, 0, 1) #we switched around the axis so make this rotate_z
            GL.glTranslatef(self.translate[0], self.translate[1],0.0)
            GL.glScale(self.scale,self.scale,self.scale)
            GL.glShadeModel(GL.GL_SMOOTH)
            GL.glLightfv(GL.GL_LIGHT0,GL.GL_POSITION, (1.,1.,0.))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.,0.,0.,1.))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (1.0,1.0,1.0,1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR,(1.,1.,1.,1.))
            GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT,(.2,.2,.2,1.0))
            GL.glEnable(GL.GL_LIGHT0)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK,GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glMaterial(GL.GL_FRONT_AND_BACK,GL.GL_SPECULAR,(1.,1.,1.,1.))
            GL.glMaterial(GL.GL_FRONT_AND_BACK,GL.GL_EMISSION, (0.,0.,0.,1.))
            

            f(self,*args,**kwargs)

            GLUT.glutSwapBuffers()
        return draw_func

def test_2d_render_func(window):
    GL.glEnable(GL.GL_POINT_SMOOTH)
    GL.glColor(0.,1.,0.)
    GL.glPointSize(20)
    GL.glBegin(GL.GL_POINTS)
    GL.glVertex(0.,0.)
    GL.glVertex(.25,.25)
    GL.glVertex(.25,-.25)
    GL.glVertex(-.25,.25)
    GL.glVertex(-.25,-.25)
    GL.glEnd()
    return

def test_3d_render_func(window):
    GL.glEnable(GL.GL_POINT_SMOOTH)
    GL.glColor(0.,1.,0.)
    GL.glPointSize(20)
    GL.glBegin(GL.GL_POINTS)
    GL.glVertex(0.,0.)
    GL.glVertex(.25,.25,.25)
    GL.glVertex(.25,.25,-.25)
    GL.glVertex(.25,-.25,.25)
    GL.glVertex(.25,-.25,-.25)
    GL.glVertex(-.25,.25,.25)
    GL.glVertex(-.25,.25,-.25)
    GL.glVertex(-.25,-.25,.25)
    GL.glVertex(-.25,-.25,-.25)
    GL.glEnd()
    return

def test_2d():
    gl_2d_window(background_color=(1.,1.,1.,1.),render_func=test_2d_render_func)

def test_3d():
    gl_3d_window(render_func=test_3d_render_func)
