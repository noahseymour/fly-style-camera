import pygame
import textwrap
import numpy as np
import math
import time
from OpenGL.GL import *
from PIL import Image 
from ctypes import sizeof, c_void_p

def load_program(vertex_source, fragment_source):
    vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_source)
    if vertex_shader == 0:
        return 0

    fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_source)
    if fragment_shader == 0:
        return 0

    program = glCreateProgram()

    if program == 0:
        return 0

    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)

    glLinkProgram(program)

    if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
        glDeleteProgram(program)
        return 0

    return program

def load_shader(shader_type, source):
    shader = glCreateShader(shader_type)

    if shader == 0:
        return 0

    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS, None) == GL_FALSE:
        info_log = glGetShaderInfoLog(shader)
        print(info_log)
        glDeleteProgram(shader)
        return 0

    return shader

def load_texture(filename):
    img = Image.open(filename, 'r').convert("RGB")
    img_data = np.array(img, dtype=np.uint8)
    w, h = img.size

    texture = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    return texture

def perspective(fovy, aspect, near, far):
    ymax = near * np.tan(fovy * np.pi / 360)
    xmax = ymax * aspect
    
    l, r, b, t, n, f = -xmax, xmax, -ymax, ymax, near, far
    
    return np.array([
        [2*n/(r-l), 0, 0, 0],
        [0, 2*n/(t-b), 0, 0],
        [(r+l)/(r-l), (t+b)/(t-b), -(f+b)/(f-n), -1],
        [0, 0, -2*f*n / (f-n), 0]
    ])

def translation(vec):
    m = np.identity(4, dtype=np.float32)
    m[3, 0:3] = vec[:3]
    return m

def rotate(angle, x, y, z):
    s = math.sin(math.radians(angle))
    c = math.cos(math.radians(angle))
    magnitude = math.sqrt(x*x + y*y + z*z)
    nc = 1 - c

    x /= magnitude
    y /= magnitude
    z /= magnitude

    return np.array([
        [     c + x**2 * nc, y * x * nc - z * s, z * x * nc + y * s, 0],
        [y * x * nc + z * s,      c + y**2 * nc, y * z * nc - x * s, 0],
        [z * x * nc - y * s, z * y * nc + x * s,      c + z**2 * nc, 0],
        [                 0,                  0,                  0, 1],
    ])

def look_at(p, t, wu):
    d = p - t
    r = np.cross(wu, d)
    u = np.cross(d, r)
    
    d /= np.linalg.norm(d)
    r /= np.linalg.norm(r)
    u /= np.linalg.norm(u)
    
    M_r = np.array([
        [r[0], r[1], r[2], 0],
        [u[0], u[1], u[2], 0],
        [d[0], d[1], d[2], 0],
        [0,    0,    0,    1]
    ])
    M_t = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-p[0], -p[1], -p[2], 1]
    ])
    return np.dot(M_t, M_r)

vertex_shader_source = textwrap.dedent("""\
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;
    layout (location = 2) in vec2 aTexCoord;

    out vec3 ourColor;
    out vec2 TexCoord;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        ourColor = aColor;
        TexCoord = aTexCoord;
    }
    """)

fragment_shader_source = textwrap.dedent("""\
    #version 330 core
    out vec4 FragColor;
    
    in vec3 ourColor;
    in vec2 TexCoord;

    uniform sampler2D ourTexture;

    void main()
    {
        FragColor = texture(ourTexture, TexCoord);
    }
    """)

vertices = [
    0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 0.0,
    0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 1.0, 0.0,
    0.5,  0.5, -0.5,  1.0, 0.0, 0.0, 1.0, 1.0,
    0.5,  0.5, -0.5,  1.0, 0.0, 0.0, 1.0, 1.0,
    -0.5,  0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
    -0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 0.0,

    -0.5, -0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,
    0.5, -0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,
    0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 1.0,
    0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 1.0,
    -0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
    -0.5, -0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,

    -0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,
    -0.5,  0.5, -0.5,  1.0, 0.0, 0.0, 1.0, 1.0,
    -0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
    -0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
    -0.5, -0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,
    -0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,

    0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,
    0.5,  0.5, -0.5,  1.0, 0.0, 0.0, 1.0, 1.0,
    0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
    0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
    0.5, -0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,
    0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,

    -0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
    0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 1.0, 1.0,
    0.5, -0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,
    0.5, -0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,
    -0.5, -0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0,

    -0.5,  0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
    0.5,  0.5, -0.5,  1.0, 0.0, 0.0, 1.0, 1.0,
    0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,
    0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 1.0, 0.0,
    -0.5,  0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,
    -0.5,  0.5, -0.5,  1.0, 0.0, 0.0, 0.0, 1.0
]
vertices = (GLfloat * len(vertices))(*vertices)

# from local space to world space, after a translation of (0, 0, -5) has been applied
cube_positions = [
    np.array([0.0, 0.0, 0.0]),
    np.array([2.0, 5.0, -15.0]),
    np.array([-1.5, -2.2, -2.5]),
    np.array([3.8, -3.5, 4.5]),
    np.array([2.4, -0.4, -3.5]),
    np.array([-1.7, 3.0, -7.5]),
    np.array([1.5, 2.0, -2.5]),
    np.array([1.5, 0.2, -1.5]),
    np.array([-3.8,-2.0, -12.3]),
    np.array([5.0, 2.4, -10.0]),
]   

width, height = 800, 600
pygame.display.set_mode((width, height), pygame.DOUBLEBUF|pygame.OPENGL)
pygame.display.set_caption("OpenGL Window")
clock = pygame.time.Clock()

glViewport(0, 0, width, height) # let opengl know how to map NDC coordinates to framebuffer coordinates

# CAMERA
camera_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
camera_speed = 0.05

pitch = 0.0
yaw = -90.0

# MATRICES
projection_matrix = perspective(45.0, 800/600, 0.1, 100.0)

program = load_program(vertex_shader_source, fragment_shader_source)

# VBO and VAO
VBO = glGenBuffers(1)
VAO = glGenVertexArrays(1)

# Bind VBO and VAO

# VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO) # now VBO links to GL_ARRAY_BUFFER
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW) # send buffer data

# VAO
glBindVertexArray(VAO)
glEnableVertexAttribArray(0)
glEnableVertexAttribArray(1)
glEnableVertexAttribArray(2)

# tell OpenGL what the vao data will be used for
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), c_void_p(0))
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), c_void_p(3 * sizeof(GLfloat)))
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), c_void_p(6 * sizeof(GLfloat)))

# Textures
texture = load_texture("container.png")

# get lock on transform matrix
modelLoc = glGetUniformLocation(program, "model")
viewLoc = glGetUniformLocation(program, "view")
projectionLoc = glGetUniformLocation(program, "projection")

# Set polygon mode
# glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

# Misc
keys = list()
dt = 0
l_f = 0

sensitivity = 0.1

# Use program
glUseProgram(program)

glEnable(GL_DEPTH_TEST)

pygame.mouse.set_visible(False)

# Event loop
while True:
    pygame.event.set_grab(True)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
            if event.key == pygame.K_w:
                keys.append('W')
            if event.key == pygame.K_s:
                keys.append('S')
            if event.key == pygame.K_a:
                keys.append('A')
            if event.key == pygame.K_d:
                keys.append('D')
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                keys.remove('W')
            if event.key == pygame.K_s:
                keys.remove('S')
            if event.key == pygame.K_a:
                keys.remove('A')
            if event.key == pygame.K_d:
                keys.remove('D')
        if event.type == pygame.MOUSEMOTION:
            x, y = event.rel
            x *= sensitivity
            y *= sensitivity
            
            yaw -= x
            pitch += y
            
            if pitch > 89.0:
                pitch = 89.0
            if pitch < -89.0:
                pitch = -89.0
                
            camera_front[0] = np.cos(math.radians(yaw)) * np.cos(math.radians(pitch))
            camera_front[1] = np.sin(math.radians(pitch))
            camera_front[2] = np.sin(math.radians(yaw)) * np.cos(math.radians(pitch))
            
            camera_front /= np.linalg.norm(camera_front)
    
    c_f = time.time()
    dt = c_f - l_f
    l_f = c_f
    camera_speed = 2.5 * dt
    
    for key in keys:
        if 'W' in keys:
            camera_pos += camera_speed * camera_front
        if 'S' in keys:
            camera_pos -= camera_speed * camera_front
        if 'A' in keys:
            dx = np.cross(camera_front, camera_up)
            dx /= np.linalg.norm(dx)
            camera_pos -= dx * camera_speed
        if 'D' in keys:
            dx = np.cross(camera_front, camera_up)
            dx /= np.linalg.norm(dx)
            camera_pos += dx * camera_speed
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
    for i in range(10):
        model_matrix = rotate(10*i + 20, 1.0, 0.3, 0.5)
        view_matrix = np.dot(translation(cube_positions[i]), look_at(camera_pos, camera_pos + camera_front, camera_up))
        
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model_matrix)
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view_matrix)
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projection_matrix)
        
        glDrawArrays(GL_TRIANGLES, 0, 36)
    
    pygame.display.flip()
    clock.tick(60)
