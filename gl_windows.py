import os
import threading
import multiprocessing


try:
    import OpenGL.GLUT as GLUT
    import OpenGL.GL as GL
    import OpenGL.GLU as GLU
except ImportError:
    raise ImportError("Need working versions of PyOpenGL with GL,GLU,and GLUT")

try:
    from PIL import Image
except:
    try:
        import Image
    except:
        raise ImportError('Cannot import suitable image libary, you will not be able to save images. Install PIL or Image')
    
class gl_window(object):
    """ General Class object to create an openGL window. Window can be
    either 2 or 3 dimensional. Creates orthographic projection with
    some sensible settings, and the best I could do for lighting. The
    class takes a function render_func which is used to update the
    display (using its appropriate arguments). This function is
    wrapped around a function used to do the translation and rotation
    and zooming appropriately. These actions are completed using the
    mouse. Left click rotates. Right click translates and the scroll
    wheel zooms. The window has several commands. Pressing the 'r' key
    will reset the display to the initial configuration. Pressing 'q'
    exits the window. pressing 's' will save the current image to the
    supplied directory with the supplied (or infered)
    extension. Finally, 'l' turns lights on and off. This only makes a
    difference if your drawing function draws objects which reflect
    light. This is included because it is slightly faster. I currently
    also include a lot of antialiasing and smoothing to make things
    look good, if it runs slow (it shouldn't) you can turn them off
    in render_func.

    Caveat: OpenGL actually blocks until the rendering loop is exited.
    This means that nothing else can run while the window is up, until
    the window is destroyed. Unfortunately if this window is spawned
    by an interpreter, there is no clean way to kill the OpenGL engine
    in such that you can open another openGL window in the same
    interpreter.  I've hacked around this by having the class
    immediately spawn a child process and run the rest of the window
    in that child process. This way you get your interpreter back
    immediately, and if running from a shell your shell back. The only
    disadvantage is that if you fork from an interpreter that is
    already bloated with memory, it could bog your machine down a bit.

    Paramters:
    ----------
    dim: int
       : Either 2 or 3 for the projection
    width: int
         : Initial window width
    height: int
          : Initial window height.
    title : String
          : Window Title
    refresh: Int
           : Number of milliseconds between refreshes
             of the window using render_func
    background_color: tuple of floats
                    : background color in RGBA format
    render_func: function
               : The function to be called at each time the screen is
                 refreshed. This should always be constructed so
                 it's first argument is window, which is a reference
                 to the openGL window, this way your rendering can
                 interact with the current window object.
    rf_args : tuple
            : arguments to be passed to render_func
    rf_kwargs : dict
              : keyword argments to be pass to render func
    save_file : string
              : Location and filename for saving. It will save files as
                savefile[i] where i increments with each save, if an
                extension is included it will attempt infer the filetype
                to save it as, otherwise it will default to 'pdf'
    save_file_type: string
                  : Format to save file in.
    thread_func : function
                  This function allows communication with the running
                  gl_window through any means. ZeroMQ sockets are great
                  It should take a gl_window as it's first argument
    **data      : dict
                  The other keyword arguments that are stored as data in
                  in the window. This allows the render func to access
                  aribitrary data in the gl_window. This is a repository,
                  so that the thread_func can recieve information and modify
                  this dictionary, and the render func can use this information.
    """


    def __init__(self,
                 dim=3,
                 width=800,
                 height=800,
                 title="",
                 refresh=15,
                 background_color = (0.,0.,0.,0.),
                 render_func=lambda window: None,
                 comm_thread=lambda window: None,
                 save_path='./',
                 save_file_type=None,
                 **data):

        self.data = data

        self._mouse_down = False
        self._mouse_old = [0., 0.]
        self._rotate = [0., 0., 0.]
        self._translate = [0., 0., 0.]
        self._scale = 1.0
        self._dim=dim
        self.title = title
        self._render_func = render_func
        self.refresh = refresh
        self.background_color = background_color
        
        # grab the file extension if its there.
        split_path_ext = os.path.splitext(save_path)
        self.save_path = split_path_ext[0]
        if save_file_type is None:
            self.save_file_ext = split_path_ext[1]
            if self.save_file_ext == '':
                self.save_file_ext = '.pdf'
        else:
            self.save_file_ext = save_file_type
        self._save_count = 0

        self._width = width
        self._height = height

        self._comm_thread = comm_thread
        
        self._glut_proc = multiprocessing.Process(target=self.glinit)
        self._glut_proc.start()
        

    def glinit(self):
        self._sock_thread = threading.Thread(target=self._comm_thread,args=(self,))
        self._sock_thread.start()
        
        GLUT.glutInit()
        GLUT.glutSetOption(GLUT.GLUT_ACTION_ON_WINDOW_CLOSE,
                           GLUT.GLUT_ACTION_GLUTMAINLOOP_RETURNS)
        # Some window attributes
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE)
        GLUT.glutInitWindowSize(self._width, self._height)
        GLUT.glutInitWindowPosition(0, 0)
        self._win = GLUT.glutCreateWindow(self.title)

        # OpenGL callbacks
        GLUT.glutDisplayFunc(self._draw(self._render_func))
        GLUT.glutKeyboardFunc(self._on_key)
        GLUT.glutMouseFunc(self._on_click)
        GLUT.glutMotionFunc(self._on_mouse_motion)
        GLUT.glutCloseFunc(self._on_exit)
        GLUT.glutTimerFunc(self.refresh, self._timer, self.refresh)

        #set up 2d or 3d viewport in nice orthographic projection
        GL.glViewport(0, 0, self._width, self._height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        if self._dim==3:
            GL.glOrtho(-1.,1.,-1,1.,-1000.,1000.)
            GL.glMatrixMode(GL.GL_MODELVIEW)
        else:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GLU.gluOrtho2D(-1.,1.,-1.,1.)
        #smooth the crap out of everything
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glEnable(GL.GL_POLYGON_SMOOTH)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE,GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT,GL.GL_NICEST)
        GL.glHint(GL.GL_POINT_SMOOTH_HINT,GL.GL_NICEST)
        GL.glHint(GL.GL_POLYGON_SMOOTH_HINT,GL.GL_NICEST)

        #Clear Everything and call the main loop
        GL.glClearColor(*self.background_color)
        self._window_open =True
        GLUT.glutMainLoop()

    # Timer callback
    def _timer(self, t):
        GLUT.glutTimerFunc(t, self._timer, t)
        GLUT.glutPostRedisplay()

    # Kill the window and the child process that spawned it.
    def _on_exit(self):
        GLUT.glutLeaveMainLoop()
        
    # handles some keys.
    def _on_key(self, *args):
        if args[0] == 'q':
            self._on_exit()
        elif args[0] == 'r':
            self._rotate = [0., 0., 0.]
            self._translate = [0., 0., 0.]
            self._scale = 1.0
        elif args[0] == 's':
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            pixel_array = GL.glReadPixels(0,0,vp[2],vp[3],GL.GL_RGB,GL.GL_UNSIGNED_BYTE)
            pilImage = Image.fromstring(mode="RGB",size=(vp[3],vp[2]),data=pixel_array)
            pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
            pilImage.save(self.save_path + str(self._save_count) + '.'  + self.save_file_ext)
            self._save_count += 1
            
    # Handle the clicks, and scale or update as necessary
    def _on_click(self, button, state, x, y):
        if state == GLUT.GLUT_DOWN:
            self._mouse_down = True
            self._button = button
            if self._button == 3 or self._button == 5:
                self._scale *= 1.1
            if self._button == 4 or self._button == 6:
                self._scale *= 0.9
        else:
            self._mouse_down = False
        self._mouse_old[0] = x
        self._mouse_old[1] = y

    # Figure out rotation and translation amounts
    def _on_mouse_motion(self, x, y):
        dx = x - self._mouse_old[0]
        dy = y - self._mouse_old[1]
        if self._mouse_down and self._button == 0: #left button
            if self._dim == 3:
                self._rotate[0] += dy * .2
                self._rotate[1] += dx * .2
            else:
                #This fixes the rotation in 2d so we always rotate
                # the way the user drags
                vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
                if x > vp[2]/2:
                    self._rotate[0] -= dy * .2
                else:
                    self._rotate[1] += dy * .2
                if y > vp[3]/2:
                    self._rotate[1] += dx * .2
                else:
                    self._rotate[1] -= dx * .2
        elif self._mouse_down and self._button == 2: #right button
            self._translate[0] += dx * .001
            self._translate[1] -= dy * .001
        self._mouse_old[0] = x
        self._mouse_old[1] = y

    ###END GL CALLBACKS


    def _draw(self,f):
        def draw_func():
            #Clear the current buffer
            GL.glFlush()
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            # Render stuff
            f(self)
            # Load the identity matrix
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

            #Transform the view in the appropriate way
            if self._dim == 3:
                GL.glTranslatef(0.,0.,-2)
                GL.glTranslatef(self._translate[0],
                                self._translate[1],
                                self._translate[2])
                GL.glRotatef(self._rotate[0], 1, 0, 0)
                GL.glRotatef(self._rotate[1], 0, 1, 0) 

            else:
                GL.glTranslatef(0.,0.,0.)
                GL.glTranslatef(self._translate[0], self._translate[1],0.0)
                GL.glRotatef(self._rotate[0], 0, 0, 1)
                GL.glRotatef(self._rotate[1], 0, 0, 1) 

            # Scale
            GL.glScale(self._scale,self._scale,self._scale)
            #More niceness
            GL.glShadeModel(GL.GL_SMOOTH)

            # Swap out the new rendering.
            GLUT.glutSwapBuffers()
        return draw_func
