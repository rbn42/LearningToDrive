#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Greeter.

Usage:
  commands.py random
  commands.py control
  commands.py -h | --help

Options:
  -h --help     Show this screen.
"""
import numpy as np
from config import random_seed

rng = np.random.RandomState(random_seed)
import scipy.misc
import math
# import cv2
# from matplotlib import pyplot as plt
from panda3d.core import loadPrcFileData
from panda3d.core import ConfigVariableString

from panda3d.bullet import BulletGhostNode
# loadPrcFileData("", "clock-mode limited")
# loadPrcFileData("", "clock-frame-rate 600")
# loadPrcFileData("", "")
#
# Bump mapping is a way of making polygonal surfaces look
# less flat.  This sample uses normal mapping for all
# surfaces, and also parallax mapping for the column.
#
# This is a tutorial to show how to do normal mapping
# in panda3d using the Shader Generator.
from direct.showbase.ShowBase import ShowBase
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionSphere, CollisionPlane, CollisionTube
from panda3d.core import CollisionBox
from panda3d.core import Plane, Vec3, Point3
from panda3d.core import CollideMask
from panda3d.core import CollisionHandlerQueue, CollisionRay
from panda3d.core import WindowProperties
from panda3d.core import Filename, Shader
from panda3d.core import AmbientLight, PointLight
# from panda3d.core import TextNode
from panda3d.core import PandaNode, NodePath, Camera, TextNode
from panda3d.core import LPoint3, LVector3
from direct.task.Task import Task
from direct.actor.Actor import Actor
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
from direct.filter.CommonFilters import *
import os
import sys
import logging

'''
我怀疑当游戏渲染窗口被遮蔽的时候会造成渲染不准确,最好尽量避免这样的情况
'''


class Demo(ShowBase):

   # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    def __init__(self, img_size=512, screen_off=True, target_area_radius=5, initial_area_radius=10, keyboard_input=False, random_reset_around_target=False, test=False):
        logging.info('random_reset_around_target :%s',
                     random_reset_around_target)
        self.random_reset_around_target = random_reset_around_target
        self.keyboard_input = keyboard_input
        # Configure the parallax mapping settings (these are just the defaults)
        self.img_size = img_size
        self.initial_area_radius = initial_area_radius
        self.target_area_radius = target_area_radius
        loadPrcFileData("", "side-by-side-stereo 1")
        if test:
            loadPrcFileData("", "load-display p3tinydisplay")
        loadPrcFileData("", "transform-cache false")
        loadPrcFileData("", "audio-library-name null")  # Prevent ALSA errors
        loadPrcFileData("", "win-size %d %d" % (2 * img_size, img_size))
        loadPrcFileData("", "parallax-mapping-samples 3\n"
                            "parallax-mapping-scale 0.1")

        if screen_off:
            # Spawn an offscreen buffer
            loadPrcFileData("", "window-type offscreen")
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)

        self.keyMap = {
            "left": 0, "right": 0, "forward": 0, "cam-left": 0, "cam-right": 0}

        # Load the 'abstract room' model.  This is a model of an
        # empty room containing a pillar, a pyramid, and a bunch
        # of exaggeratedly bumpy textures.

        self.room = loader.loadModel("models/abstractroom")
        self.room.reparentTo(render)

        # Create the main character, Ralph

        ghost = BulletGhostNode('Ghost')
        ghostNP = render.attachNewNode(ghost)
      #  self.agent = Actor("models/agent",
      #                     {"run": "models/agent-run",
      #                      "walk": "models/agent-walk"})
        self.agent = ghostNP
        self.agent.reparentTo(render)
#        self.agent.setScale(.2)
        target = BulletGhostNode('target')
        self.navigation_target = render.attachNewNode(target)
        self.navigation_target.reparentTo(render)

        # knghit=Knight((0,0,0),(0.3,.3,.3,1))
        self.pieces = [Piece(self.room) for _ in range(200)]
        ##################################################
        cnodePath = self.room.attachNewNode(CollisionNode('room'))
        plane = CollisionPlane(Plane(Vec3(1, 0, 0), Point3(-60, 0, 0)))  # left
        cnodePath.node().addSolid(plane)
        plane = CollisionPlane(
            Plane(Vec3(-1, 0, 0), Point3(60, 0, 0)))  # right
        cnodePath.node().addSolid(plane)
        plane = CollisionPlane(Plane(Vec3(0, 1, 0), Point3(0, -60, 0)))  # back
        cnodePath.node().addSolid(plane)
        plane = CollisionPlane(
            Plane(Vec3(0, -1, 0), Point3(0, 60, 0)))  # front
        cnodePath.node().addSolid(plane)

        sphere = CollisionSphere(-25, -25, 0, 12.5)
        # tube = CollisionTube(-25, -25,0 , -25, -25, 1, 12.5)
        cnodePath.node().addSolid(sphere)

        box = CollisionBox(Point3(5, 5, 0), Point3(45, 45, 10))
        cnodePath.node().addSolid(box)

      #  cnodePath.show()

        # Make the mouse invisible, turn off normal mouse controls
        self.disableMouse()
        # props = WindowProperties()
        # props.setCursorHidden(True)
        # self.win.requestProperties(props)
        # self.camLens.setFov(60)
        self.camLens.setFov(80)

        # Set the current viewing target
        self.focus = LVector3(55, -55, 20)
        self.heading = 180
        self.pitch = 0
        self.mousex = 0
        self.mousey = 0
        self.last = 0
        self.mousebtn = [0, 0, 0]

        # Start the camera control task:
        # taskMgr.add(self.controlCamera, "camera-task")
        # self.accept("escape", sys.exit, [0])
        # self.accept("enter", self.toggleShader)
        # self.accept("j", self.rotateLight, [-1])
        # self.accept("k", self.rotateLight, [1])
        # self.accept("arrow_left", self.rotateCam, [-1])
        # self.accept("arrow_right", self.rotateCam, [1])

        # Accept the control keys for movement and rotation

        self.accept("escape", sys.exit)
        self.accept("arrow_left", self.setKey, ["left", True])
        self.accept("arrow_right", self.setKey, ["right", True])
        self.accept("arrow_up", self.setKey, ["forward", True])
        self.accept("a", self.setKey, ["cam-left", True])
        self.accept("s", self.setKey, ["cam-right", True])
        self.accept("arrow_left-up", self.setKey, ["left", False])
        self.accept("arrow_right-up", self.setKey, ["right", False])
        self.accept("arrow_up-up", self.setKey, ["forward", False])
        self.accept("a-up", self.setKey, ["cam-left", False])
        self.accept("s-up", self.setKey, ["cam-right", False])
        # Add a light to the scene.
        self.lightpivot = render.attachNewNode("lightpivot")
        self.lightpivot.setPos(0, 0, 25)
        self.lightpivot.hprInterval(10, LPoint3(360, 0, 0)).loop()
        plight = PointLight('plight')
        plight.setColor((5, 5, 5, 1))
        plight.setAttenuation(LVector3(0.7, 0.05, 0))
        plnp = self.lightpivot.attachNewNode(plight)
        plnp.setPos(45, 0, 0)
        self.room.setLight(plnp)

        # Add an ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = render.attachNewNode(alight)
        self.room.setLight(alnp)

        # Create a sphere to denote the light
        sphere = loader.loadModel("models/icosphere")
        sphere.reparentTo(plnp)

#         self.cameraModel = self.agent

#        self.win2 = self.openWindow()
#        self.win2.removeAllDisplayRegions()
#        self.dr2 = self.win2.makeDisplayRegion()
#        camNode = Camera('cam')
#        camNP = NodePath(camNode)
#        camNP.reparentTo(self.cameraModel)
#        camNP.setZ(150)
#        camNP.lookAt(self.cameraModel)
#        self.dr2.setCamera(camNP)
#        self.cam2 = camNP  # Camera('cam')p

        # We will detect the height of the terrain by creating a collision
        # ray and casting it downward toward the terrain.  One ray will
        # start above agent's head, and the other will start above the camera.
        # A ray may hit the terrain, or it may hit a rock or a tree.  If it
        # hits the terrain, we can detect the height.  If it hits anything
        # else, we rule that the move is illegal.
        self.cTrav = CollisionTraverser()

        cs = CollisionSphere(0, 0, 0, 0.2)
        cnodePath = self.agent.attachNewNode(CollisionNode('agent'))
        cnodePath.node().addSolid(cs)

      #  cnodePath.show()
        self.ralphGroundHandler = CollisionHandlerQueue()
        self.cTrav.addCollider(cnodePath, self.ralphGroundHandler)

        cnodePath = self.navigation_target.attachNewNode(
            CollisionNode('target'))
        cnodePath.node().addSolid(CollisionSphere(0, 0, 0, 2))
        self.cTrav.addCollider(cnodePath, self.ralphGroundHandler)

        # Tell Panda that it should generate shaders performing per-pixel
        # lighting for the room.
        self.room.setShaderAuto()

        self.shaderenable = 1

        # tex = self.win.getScreenshot()
        # tex.setFormat(Texture.FDepthComponent)

        tex = Texture()
        self.depthmap = tex
        tex.setFormat(Texture.FDepthComponent)
        altBuffer = self.win.makeTextureBuffer(
            "hello", img_size, img_size, tex, True)
        self.altBuffer = altBuffer
        altCam = self.makeCamera(altBuffer)
        altCam.reparentTo(self.agent)  # altRender)
        altCam.setZ(0.4)
        l = altCam.node().getLens()
        l.setFov(80)
        l.setNear(.1)

        camera.reparentTo(self.agent)
        camera.setZ(0.4)
        l = self.camLens
        # l.setFov(80)
        l.setNear(.1)
        # end init

    def setKey(self, key, value):
        self.keyMap[key] = value

    def step(self, action=(0, 0, 0)):
        h, p, forward = action
        self.agent.setH(self.agent, h)
        self.agent.setP(self.agent, p)
        self.agent.setY(self.agent, forward)
        relative_target_position = self.__get_relative_target_position()
        if not None == self.relative_target_position:
            speed = abs(relative_target_position) - \
                abs(self.relative_target_position)
            assert abs(speed) < 0.02 * 25 * 0.2 * 1.001 * 4
        self.relative_target_position = relative_target_position

        return len(self.get_collision_list()) > 0
        dt = 0.02  # 控制跳帧,这个问题在转移到现实的时候需要考虑,摄影机如果能提供固定的拍摄频率就最好了,
        if len(self.pieces) > 0:
            for _ in range(5):
                rng.choice(self.pieces).step(dt)

    def get_collision_list(self):
        if self.keyboard_input == False:
            f = taskMgr.getAllTasks()[-1].getFunction()
            f(None)
            f = taskMgr.getAllTasks()[2].getFunction()
            f(None)
        else:
            taskMgr.step()
        return (self.ralphGroundHandler.getEntries())
    relative_target_position = None

    def __get_relative_target_position(self):
        from_hpr = self.agent.getHpr()
        h = from_hpr[0] * math.pi / 180

        from_pos = self.agent.getPos()
        from_pos = from_pos[0] + from_pos[1] * 1j

        target_pos = self.navigation_target.getPos()
        target_pos = target_pos[0] + target_pos[1] * 1j

        relative_pos = target_pos - from_pos
        rpos = relative_pos * np.exp(-h * 1j)
        return rpos

    def get_agent_position(self):
        return self.agent.getPos()

    def get_obstacle_positions(self):
        return [piece.knight.getPos() for piece in self.pieces]

    def get_obstacle_radius(self):
        return [piece.radius for piece in self.pieces]

    def get_target_position(self):
        return self.navigation_target.getPos()

    def resetGame(self):

        self.renderFrame()
        img = self.getScreen()

        for p in self.pieces:
            p.putaway()

        self.agent.setPos((20, 20, 1000))
        size = 60

        while True:
            x, y = rng.uniform(-size, size, 2)  # rng.rand() *
            self.navigation_target.setPos((x, y, 0))
            l = self.get_collision_list()
            if len(l) < 1:
                break
        # print(i)
        target_pos = self.navigation_target.getPos()
        target_pos = target_pos[0] + target_pos[1] * 1j

        while True:
            if not self.random_reset_around_target:  # :
                x, y = rng.uniform(-size, size, 2)  # rng.rand
            else:
                while True:
                    pos = (10 + size * math.pi * rng.rand()) * \
                        np.exp(rng.rand() * math.pi * 1j)
                    pos += target_pos
                    if -size < pos.real < size and -size < pos.imag < size:
                        x, y = pos.real, pos.imag
                        break
            self.agent.setPos((x, y, 0))
            l = self.get_collision_list()
            if len(l) < 1:
                break

        self.agent.setH(rng.uniform(0, 360))
        self.relative_target_position = self.__get_relative_target_position()

        for p in self.pieces:
            p.reset(avoids=[(self.agent.getPos(), self.initial_area_radius),
                            (self.navigation_target.getPos(), self.target_area_radius)])
        # print(i)

    def renderFrame(self):
        base.graphicsEngine.renderFrame()

    def getScreen(self):
        #    base.graphicsEngine.renderFrame()
        # self.screenshot(namePrefix=self._screenshot)
        tex = self.win.getScreenshot()
        r = tex.getRamImage()
        l = np.asarray(r)
        x = tex.getXSize()
        y = tex.getYSize()
      #  print((x,y,len(l)))
        assert x * y * 4 == len(l)
        if x * y * 3 == len(l):
            l = l.reshape((y, x, 3))
        elif x * y * 4 == len(l):
            l = l.reshape((y, x, 4))
        elif x * y == len(l):
            l = l.reshape((y, x))
        else:
            a = 1 / 0
        l = np.flipud(l)
        return l

    def getDepthMap(self, p):
        # print(p.dtype)
        # p=p[:,:,:2]
        # img =p# cv2.imread(p, 0)
        img = np.mean(p, axis=2, dtype='uint8')
      #  print(img.shape)
      #  img=p[:,:,3]
      #  print(img.shape)
        # os.remove(p)
        img1 = img[:, :self.img_size]
        img2 = img[:, self.img_size:]
        disparity = self.stereo.compute(img1, img2)

        if rng.rand() < 0.1:
            p = '/dev/shm/z_depthsample%d.jpg'
            p2 = '/dev/shm/z_depthsample%d_.jpg'
            for i in range(50):
                if not os.path.exists(p % i):
                    scipy.misc.imsave(p % i, disparity)
                    scipy.misc.imsave(p2 % i, np.asarray(
                        [img1, img2, disparity]))
                    break

        return disparity, img1, img2

    def getRawImage(self, p):
        img = p  # scipy.misc.imread(p)
        # os.remove(p)
        img1 = img[:, :self.img_size]
        img = np.mean(img1, axis=2)
        return img

    def getRawImageStereo(self, img):
        img = np.mean(img, axis=2)
        img1 = img[:, :self.img_size]
        img2 = img[:, self.img_size:]
        return np.asarray([img1, img2])

    def getDepthMapT(self):
        tex = self.depthmap
        # tex = self.altBuffer.getTexture()
        r = tex.getRamImage()
        s = r.getData()
        i = np.fromstring(s, dtype='float32')
        i = i.reshape((self.img_size, self.img_size))
        i = np.flipud(i)
        return i

    def get1Ddepth(self, dmap):
        i = int(self.img_size * 0.525)  # horizon
        return dmap[i]


class Piece:

    def __init__(self, room, roomsize=60, speed_max=1, acceleration_max=1):
        self.roomsize = roomsize
        self.speed_max = speed_max
        self.acceleration_max = acceleration_max

        p = 'models/knight', "models/pawn", "models/king", "models/queen", "models/bishop", "models/knight", "models/rook"
        p = rng.choice(p)
        self.knight = loader.loadModel(p)

       # color =
        tex = loader.loadTexture('./models/wood.jpg')
        self.knight.setTexture(tex)

        self.knight.reparentTo(room)
        self.model = self.knight

        cs = CollisionSphere(0, 0, 0, 0.27)
        cnodePath = self.knight.attachNewNode(CollisionNode('piece'))
        cnodePath.node().addSolid(cs)

        self.reset()

    def reset(self, avoids=[]):
        self._speed = np.zeros(2)
        while True:
            # rng.rand() * 2 * self.roomsize - self.roomsize        y =
            # rng.rand() * 2 * self.roomsize - self.roomsize
            x, y = rng.uniform(-self.roomsize, self.roomsize, 2)
            piece_scale = rng.uniform(2, 11)  # 9 * rng.rand() + 2
            for point, radius in avoids:
                r = radius + 0.27 * piece_scale * 2
                _x, _y, _z = point
                if (_x - x)**2 + (_y - y)**2 < r**2:
                    break
            else:
                break
        self._pos = np.asarray([x, y])
        k_pos = x, y, 0
        self.knight.setPos(k_pos)

        # self.knight.reparentTo(render)
        # self.knight.setColor(color)
        self.knight.setScale(piece_scale)
        self.radius = 0.27 * piece_scale

    def putaway(self):
        self.knight.setPos((1000, 1000, 1000))

    def step(self, dt):
        # * (2 * rng.rand() - 1)        dy = self.acceleration_max * (2 * rng.rand() - 1)
        dx, dy = rng.uniform(- self.acceleration_max, self.acceleration_max, 2)
        self._speed += dx * dt, dy * dt
        norm = np.linalg.norm(self._speed)
        if norm > self.speed_max:
            self._speed *= self.speed_max / norm
        self._pos += self._speed * dt
        if self._pos[0] > self.roomsize:
            self._pos[0] = self.roomsize
            self._speed[0] = -abs(self._speed[0])
        if self._pos[0] < -self.roomsize:
            self._pos[0] = -self.roomsize
            self._speed[0] = abs(self._speed[0])
        if self._pos[1] > self.roomsize:
            self._pos[1] = self.roomsize
            self._speed[1] = -abs(self._speed[1])
        if self._pos[1] < -self.roomsize:
            self._pos[1] = -self.roomsize
            self._speed[1] = abs(self._speed[1])
        self.model.setPos((self._pos[0], self._pos[1], 0))


if '__main__' == __name__:
    from docopt import docopt
    arguments = docopt(__doc__)

    if arguments['random']:
        demo = Demo(screen_off=False, keyboard_input=False)
    elif arguments['control']:
        demo = Demo(screen_off=False, keyboard_input=True)

    self = demo
#    mywin3 = self.openWindow()
# #    from direct.gui.OnscreenImage import OnscreenImage
# #    imageObject = OnscreenImage(image='myImage.jpg', pos=(0, 0, 0))
# #    win2Render2d = NodePath('win2Render2d')
# #    win2Cam2d = base.makeCamera2d(mywin3)
# #    win2Cam2d.reparentTo(win2Render2d)
# #    imageObject.reparentTo(win2Render2d)
    p_depthmap = '/dev/shm/s_depth.npy'
    p_screen = '/dev/shm/s_screen.npy'
    p_stereo = '/dev/shm/s_stereo.npy'
    while True:
        print(demo.agent.getPos())
   #     print(demo.relative_target_position)
        action_options = [(0, 0, 0.1), (2, 0, 0.1),
                          (-2, 0, 0.1), (0, 0, 0)]  # (0,0,0)]
        if self.keyMap["left"]:
            action = action_options[1]
        elif self.keyMap["right"]:
            action = action_options[2]
        elif self.keyMap["forward"]:
            action = action_options[0]
        else:
            action = action_options[3]
        if arguments['random']:
            action = action_options[rng.randint(len(action_options))]
            # c,p=demo.step(action)
        if demo.step(action):
            demo.resetGame()
        if arguments['random']:
            base.graphicsEngine.renderFrame()
        elif arguments['control']:
            taskMgr.step()
        if not os.path.exists(p_depthmap):
            img = self.getDepthMapT()
            np.save(p_depthmap, img)
        if not os.path.exists(p_screen):
            img = self.getScreen()
            np.save(p_screen, img)
        if not os.path.exists(p_stereo):
            img = self.getScreen()
            img = self.getRawImageStereo(img)
            np.save(p_stereo, img)

#        if random.random() < 0.03:
#            i = self.getDepthMapT()
#            p = '/dev/shm/111.jpg'
#       #     if os.path.exists(p):
#       #         return
#            scipy.misc.imsave(p, i)
#            pass
# #        self.cTrav.traverse(self.cnodePath)
# #        p = getSreen()
# #        disparity = getDepthMap(p)
# #        scipy.misc.imsave(p, disparity)
# #        imageObject.setImage(p)
# #        os.remove(p)
