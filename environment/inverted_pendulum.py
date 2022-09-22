#####################################################################################
'''
Swing-up control of pendulum on a cart (RL Task)

Author  : Jongchan Baek
Date    : 2022.08.26
Contact : paekgga@postech.ac.kr
'''
#####################################################################################

import numpy as np
from math import sin, cos
import mujoco
#import mujoco_viewer
import glfw, time

class InvertedPendulumSwing:

    def __init__(self):

        # Physical parameters
        self.g = 9.81
        self.l = 0.4
        self.m = 0.291
        self.a = 0.147
        self.I = 0.0013
        self.c = 0.00154756
        self.dt = 0.02

        self.frame_skip = 10
        self.sample_time = self.dt/self.frame_skip
        self.state_dim = 5
        self.action_dim = 1
        self.action_max = 20.0
        self.pos_max = 0.45
        self.viewer = None
        self.eqi_idx = [0,1,3,4]
        self.reg_idx = [2]

        self.epi_step = 0
        self._max_episode_steps = 1000

    def reset(self):
        self.epi_step = 0
        q = np.zeros(2)
        q[0] = 0.01 * np.random.randn()
        q[1] = np.pi + .01 * np.random.randn()
        qd = .01 * np.random.randn(2)
        self.x = np.concatenate([q, qd])
        obs = self._get_obs()
        if self.viewer is not None:
            self.viewer.graph_reset()
        return obs

    def step(self, action):
        self.epi_step += 1
        act = np.asscalar(action)
        self._do_simulation(act)
        obs = self._get_obs()
        pos, cos_th, th_dot = obs[0], obs[2], obs[4]
        notdone = np.isfinite(obs).all() and (np.abs(pos) <= self.pos_max)
        notdone = notdone and np.abs(th_dot) < 27.
        r_pos = 0.8 + 0.2 * np.exp(-0.67 * pos**2)
        r_act = 0.8 + 0.2 * np.maximum(1 - (act / self.action_max)**2, 0.0)
        r_angle = 0.5 + 0.5 * cos_th
        r_vel = 0.5 + 0.5 * np.exp(-0.2 * th_dot**2)
        r = r_pos * r_act * r_angle * r_vel
        done = not notdone
        if self.epi_step == self._max_episode_steps:
            done = True
        return obs, r, done, {}

    def random_sample(self):
        return np.random.uniform(-1., 1., [self.action_dim])

    def render(self, mouse_control=False):
        if self.viewer is None:
            self._viewer_setup()
        if not self.viewer.is_alive:
            self._viewer_reset()
        if not mouse_control:
            qpos = [self.x[0], -self.x[1]+np.pi]
            qvel = [self.x[2], -self.x[3]]
            self.data.qpos = qpos
            self.data.qvel = qvel
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def _get_obs(self):
        return np.array([self.x[0], sin(self.x[1]), cos(self.x[1]), self.x[2], self.x[3]])

    def _do_simulation(self, a):
        for i in range(self.frame_skip):
            xd1 = self._derivative(self.x, a)
            xd2 = self._derivative(self.x + (self.sample_time / 2) * xd1, a)
            xd3 = self._derivative(self.x + (self.sample_time / 2) * xd2, a)
            xd4 = self._derivative(self.x + self.sample_time * xd3, a)
            xd = (xd1 + 2 * xd2 + 2 * xd3 + xd4) / 6
            self.x += self.sample_time * xd

    def _derivative(self, x, a):
        pos, th, pos_d, th_d = x
        pos_dd = a
        th_dd = (self.m*self.g*self.l*sin(th) - self.c*th_d -
                 self.m*self.l*cos(th)*pos_dd)/(self.I + self.m*self.l**2)
        x_d = np.array([pos_d, th_d, pos_dd, th_dd])
        return x_d

    def _viewer_setup(self):
        self.model = mujoco.MjModel.from_xml_path('./environment/mujoco/assets/sip.xml')
        self.data = mujoco.MjData(self.model)
        self._viewer_reset()

    def _viewer_reset(self):
        self.viewer = Viewer(model=self.model, data=self.data,
                             width=1000, height=600,
                             title='InvertedPendulumSwing',
                             hide_menus=True)
        self.viewer.cam.distance = self.model.stat.extent * 1.35
        self.viewer.cam.lookat[2] -= 0
        self.viewer.cam.elevation += 35
        self.viewer.cam.azimuth = 115

'''
class Viewer(mujoco_viewer.MujocoViewer):

    def __init__(self, width, height, **kwargs):
        super().__init__(width=width, height=height, **kwargs)
        self.width, self.height = width, height
        self.n_timestep = 1000
        self.graph_reset()

    def graph_reset(self):
        fig1, fig_viewport1 = self._set_figure(0, self.viewport.height)
        fig2, fig_viewport2 = self._set_figure(0, 2*int(self.viewport.height/4))
        fig3, fig_viewport3 = self._set_figure(0, 3*int(self.viewport.height/4))
        fig4, fig_viewport4 = self._set_figure(0, 0)
        self.fig = [fig1, fig2, fig3, fig4]
        self.fig_viewport = [fig_viewport1, fig_viewport2, fig_viewport3, fig_viewport4]
        fig1.range[1][0] = -0.5
        fig1.range[1][1] = 0.5
        fig1.linergb = [1, 1, 0]
        fig1.flg_ticklabel[0] = 0
        fig1.linename[0] = 'CartPos.'
        fig2.range[1][0] = -4.0
        fig2.range[1][1] = 4.0
        fig2.linergb = [1, 1, 1]
        fig2.linename[0] = 'Ang.'
        fig2.flg_ticklabel[0] = 0
        fig3.range[1][0] = -3.
        fig3.range[1][1] = 3.
        fig3.linergb = [1, 1, 0]
        fig3.linename[0] = 'CartVel.'
        fig3.flg_ticklabel[0] = 0
        fig4.range[1][0] = -30.
        fig4.range[1][1] = 30.
        fig4.linergb = [1, 1, 1]
        fig4.linename[0] = 'Ang.Vel.'
        fig4.flg_ticklabel[0] = 0

    def _set_figure(self, loc_x, loc_y):
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        for i in range(0, self.n_timestep):
            fig.linedata[0][2 * i] = float(-i)
        fig_viewport = mujoco.MjrRect(loc_x, loc_y, 200, int(self.viewport.height/4))
        mujoco.mjr_figure(fig_viewport, fig, self.ctx)
        fig.flg_extend = 1
        fig.flg_symmetric = 0
        fig.flg_legend = 1
        fig.range[0][1] = 0.01
        fig.gridsize = [2, 5]
        fig.legendrgba = [0, 0, 0, 0]
        fig.figurergba = [0, 0, 0, 0.8]
        fig.panergba = [0, 0, 0, 0.5]
        return fig, fig_viewport

    def _sensorupdate(self):
        sensor_data = (self.data.qpos + np.array([0.,-np.pi]))*np.array([1.,-1.])
        sensor_data = np.concatenate([sensor_data, self.data.qvel])
        for i in range(len(self.fig)):
            pnt = int(mujoco.mju_min(self.n_timestep, self.fig[i].linepnt[0] + 1))
            for j in range(pnt-1, 0, -1):
                self.fig[i].linedata[0][2 * j + 1] = self.fig[i].linedata[0][2 * j - 1]
            self.fig[i].linepnt[0] = pnt
            self.fig[i].linedata[0][1] = sensor_data[i]

    def _update_graph_size(self):
        for i in range(len(self.fig_viewport)):
            self.fig_viewport[i].left = 0
            self.fig_viewport[i].bottom = int((3-i)*self.viewport.height/4)
            self.fig_viewport[i].width = int(0.3*self.viewport.width)
            self.fig_viewport[i].height = int(self.viewport.height/4)+1

    def render(self):
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window)
            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                # overlay items
                for gridpos, [t1, t2] in self._overlay.items():
                    menu_positions = [mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      mujoco.mjtGridPos.mjGRID_BOTTOMLEFT]
                    if gridpos in menu_positions and self._hide_menus:
                        continue

                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.ctx)

                if not self._paused:
                    self._sensorupdate()
                    self._update_graph_size()
                    for fig, viewport in zip(self.fig, self.fig_viewport):
                        mujoco.mjr_figure(viewport, fig, self.ctx)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + \
                0.1 * (time.time() - render_start)

            # clear overlay
            self._overlay.clear()

        if self._paused:
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1
'''