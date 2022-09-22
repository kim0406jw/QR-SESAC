import numpy as np
from math import sin, cos
import mujoco
#import mujoco_viewer
import glfw, time
from copy import deepcopy

class TripleInvertedPendulumSwing:

    def __init__(self):
        self.sample_time = 0.002
        self.frame_skip = 10
        self.control_frequency = self.sample_time * self.frame_skip
        self.rail_length = 0.9
        self.action_max = 10.0
        self.action_dim = 1
        self.n_history = 3
        self.state_dim = 11 * self.n_history
        self.viewer = None
        self.eqi_idx = np.array([[0,1,2,3,7,8,9,10]]*self.n_history)
        self.eqi_idx += 11*np.arange(self.n_history).reshape([-1,1])
        self.eqi_idx = self.eqi_idx.reshape([-1]).tolist()
        self.reg_idx = np.array([[4,5,6]]*self.n_history)
        self.reg_idx += 11*np.arange(self.n_history).reshape([-1,1])
        self.reg_idx = self.reg_idx.reshape([-1]).tolist()

        self._max_episode_steps = 1000
        self.epi_step = 0

    def step(self, a):
        self.epi_step += 1
        self._do_simulation(a[0])
        ob = self._get_obs()
        pos, cos_th1, cos_th2, cos_th3 = ob[0], ob[4], ob[5], ob[6]
        thd1, thd2, thd3 = ob[8], ob[9], ob[10]
        notdone = np.isfinite(ob).all() and (np.abs(ob[0]) <= self.rail_length/2)
        notdone = notdone and np.all(np.abs(ob[-3:])<27.)
        r_pos = 0.8 + 0.2*np.exp(-0.67*pos**2)
        r_act = 0.8 + 0.2 * np.maximum(1 - (a / self.action_max) ** 2, 0.0)
        r_angle1 = 0.5 + 0.5 * cos_th1
        r_angle2 = 0.5 + 0.5 * cos_th2
        r_angle3 = 0.5 + 0.5 * cos_th3
        r_angle = r_angle1 * r_angle2 * r_angle3
        r_vel1 = 0.5 + 0.5 * np.exp(-0.1 * thd1**2)
        r_vel2 = 0.5 + 0.5 * np.exp(-0.1 * thd2**2)
        r_vel3 = 0.5 + 0.5 * np.exp(-0.1 * thd3**2)
        r_vel = min(r_vel1, min(r_vel2, r_vel3))
        r = r_pos * r_act * r_angle * r_vel

        done = not notdone
        ob = np.reshape(ob, [1, -1])
        self.observ = np.hstack([ob] + [self.observ[0, :11 * (self.n_history - 1)].reshape([1, -1])])

        true_done = float(done)
        if self.epi_step == self._max_episode_steps:
            done = True

        return self.observ.copy()[0], r[0], done, true_done

    def reset(self):
        self.epi_step = 0
        q = np.zeros(4)     #[x, th1, th2, th3]
        q[0] = 0.01*np.random.randn()
        q[1] = np.pi + .01*np.random.randn()
        q[2] = np.pi + .1*np.random.randn()
        q[3] = np.pi + .1*np.random.randn()
        qd = .01*np.random.randn(4)
        self.x = np.concatenate([q,qd])
        ob = self._get_obs()
        if self.viewer is not None:
            self.viewer.graph_reset()
        init_obs = np.reshape(ob, [1, -1])
        self.observ = np.hstack([init_obs] * self.n_history)
        return self.observ.copy()[0]

    def test_reset(self, idx):
        self.epi_step = 0
        if idx % 2 == 0:
            q = np.zeros(4)     #[x, th1, th2, th3]
            q[0] = 0.2
            q[1] = np.pi + 1.0
            q[2] = np.pi + 1.0
            q[3] = np.pi + 1.0
            qd = np.zeros(4)
        else:
            q = np.zeros(4)  # [x, th1, th2, th3]
            q[0] = -0.2
            q[1] = np.pi - 1.0
            q[2] = np.pi - 1.0
            q[3] = np.pi - 1.0
            qd = np.zeros(4)
        self.x = np.concatenate([q,qd])
        ob = self._get_obs()
        if self.viewer is not None:
            self.viewer.graph_reset()
        init_obs = np.reshape(ob, [1, -1])
        self.observ = np.hstack([init_obs] * self.n_history)
        return self.observ.copy()[0]

    def random_sample(self):
        return np.random.uniform(-1.,1.,[self.action_dim])

    def _do_simulation(self, a):
        for i in range(self.frame_skip):
            acc = self._v2a(a)
            xd1 = self._derivative(self.x, acc)
            xd2 = self._derivative(self.x + (self.sample_time/2)*xd1, acc)
            xd3 = self._derivative(self.x + (self.sample_time/2)*xd2, acc)
            xd4 = self._derivative(self.x + self.sample_time*xd3, acc)
            xd = (xd1 + 2*xd2 + 2*xd3 + xd4)/6
            self.x += self.sample_time*xd

    def _get_obs(self):
        th1 = self.x[1]
        th2 = self.x[2] - self.x[1]
        th3 = self.x[3] - self.x[2]
        return np.array([self.x[0],sin(th1),sin(th2),sin(th3),
                         cos(th1), cos(th2), cos(th3), self.x[4],
                         self.x[5], self.x[6]-self.x[5], self.x[7]-self.x[6]])

    def _derivative(self, x, a):

        l1, l2, l3 = 0.1645, 0.210, 0.245
        a1 = 0.073155254929649
        a2 = 0.118284040031105
        a3 = 0.151746102334071
        m1 = 0.260382999723777
        m2 = 0.160625541146578
        m3 = 0.144946326257173
        J1 = 0.001395106789561
        J2 = 0.001283738784734
        J3 = 0.001539479408814
        d1 = 0.0006812938122204569
        d2 = 0.0004237426030179191
        d3 = 0.0005178578106688010
        g = 9.81

        q, qd = x[:4], x[4:]

        m11 = J1 + (a1 ** 2) * m1 + (l1 ** 2) * (m2 + m3)
        m12 = a2 * l1 * m2 * cos(q[1] - q[2]) + l1 * l2 * m3 * cos(q[1] - q[2])
        m13 = a3 * l1 * m3 * cos(q[1] - q[3])
        m21 = a2 * l1 * m2 * cos(q[1] - q[2]) + l1 * l2 * m3 * cos(q[1] - q[2])
        m22 = J2 + (a2 ** 2) * m2 + (l2 ** 2) * m3
        m23 = a3 * l2 * m3 * cos(q[2] - q[3])
        m31 = a3 * l1 * m3 * cos(q[1] - q[3])
        m32 = a3 * l2 * m3 * cos(q[2] - q[3])
        m33 = J3 + (a3 ** 2) * m3
        M = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])

        c11 = c22 = c33 = 0
        c12 = qd[2] * a2 * l1 * m2 * sin(q[1] - q[2]) + qd[2] * l1 * l2 * m3 * sin(q[1] - q[2])
        c13 = qd[3] * a3 * l1 * m3 * sin(q[1] - q[3])
        c21 = -qd[1] * a2 * l1 * m2 * sin(q[1] - q[2]) - qd[1] * l1 * l2 * m3 * sin(q[1] - q[2])
        c23 = qd[3] * a3 * l2 * m3 * sin(q[2] - q[3])
        c31 = -qd[1] * a3 * l1 * m3 * sin(q[1] - q[3])
        c32 = -qd[2] * a3 * l2 * m3 * sin(q[2] - q[3])
        C = np.array([[c11, c12, c13],
                      [c21, c22, c23],
                      [c31, c32, c33]])

        d11 = d1 + d2
        d12 = -d2
        d13 = 0
        d21 = -d2
        d22 = d2 + d3
        d23 = -d3
        d31 = 0
        d32 = -d3
        d33 = d3
        D = np.array([[d11, d12, d13],
                      [d21, d22, d23],
                      [d31, d32, d33]])

        g11 = -g * (a1 * m1 + l1 * m2 + l1 * m3) * sin(q[1])
        g21 = -g * (a2 * m2 + l2 * m3) * sin(q[2])
        g31 = -a3 * g * m3 * sin(q[3])
        G = np.array([g11, g21, g31]).T

        b11 = a1 * m2 * cos(q[1]) + l1 * m2 * cos(q[1]) + l1 * m3 * cos(q[1])
        b21 = a2 * m2 * cos(q[2]) + l2 * m3 * cos(q[2])
        b31 = a3 * m3 * cos(q[3])
        B = np.array([b11, b21, b31]).T

        Minv = np.linalg.inv(M)
        F0 = -Minv @ (C @ qd[1:].T + D @ qd[1:].T + G)
        F1 = Minv @ B

        fx = np.concatenate([qd, np.zeros(1), F0.T])
        gx = np.concatenate([np.zeros(4), np.ones(1), F1.T])
        xd = fx + gx * a
        return xd

    def _v2a(self, v):
        xdot = deepcopy(self.x[-4])
        k = 0.06
        J = 0.000033
        R = 0.8
        r = 0.016
        b = 0.00035
        acc = (k*r*v - (k*k+b*R)*xdot)/(J*R)
        return acc

    # def _derivative(self, x, f):
    #
    #     l1 = 0.1645; l2 = 0.210; l3 = 0.245
    #     a1 = 0.077792803144923; a2 = 0.118304871010206; a3 = 0.142848166462599
    #     m0 = 0.485; m1 = 0.209031904484509; m2 = 0.114999198087930; m3 = 0.146790233805106
    #     J1 = 0.001145259713644; J2 = 0.0008050827698029238; J3 = 0.001600887002439
    #     d0 = 0.4524; d1 = 0.001547562611052; d2 = 0.0008598850094355782; d3 = 0.0006480318611403285
    #     g = 9.81
    #
    #     q, qd = x[:4], x[4:]
    #
    #     m11 = m0 + m1 + m2 + m3
    #     m12 = m21 = -(a1*m1 + l1*m2 + l1*m3)*cos(q[1])
    #     m13 = m31 = -(a2*m2 + l2*m3)*cos(q[2])
    #     m14 = m41 = -a3*m3*cos(q[3])
    #     m22 = J1 + a1*a1*m1 + l1*l1*m2 + l1*l1*m3
    #     m23 = m32 = (a2*l1*m2 + l1*l2*m3)*cos(q[1]-q[2])
    #     m24 = m42 = a3*l1*m3*cos(q[1]-q[3])
    #     m33 = J2 + a2*a2*m2 + l2*l2*m3
    #     m34 = m43 = a3*l2*m3*cos(q[2]-q[3])
    #     m44 = J3 + a3*a3*m3
    #     M = np.array([[m11,m12,m13,m14],
    #                   [m21,m22,m23,m24],
    #                   [m31,m32,m33,m34],
    #                   [m41,m42,m43,m44]])
    #
    #     c11 = c22 = c33 = c44 = c21 = c31 = c41 = 0
    #     c12 = qd[1]*sin(q[1])*(a1*m1 + l1*m2 + l1*m3)
    #     c13 = qd[2]*sin(q[2])*(a2*m2 + l2*m3)
    #     c14 = qd[3]*a3*m3*sin(q[3])
    #     c23 = qd[2]*sin(q[1]-q[2])*(a2*l1*m2 + l1*l2*m3)
    #     c24 = qd[3]*sin(q[1]-q[3])*a3*l1*m3
    #     c32 = -qd[1]*sin(q[1]-q[2])*(a2*l1*m2 + l1*l2*m3)
    #     c34 = qd[3]*sin(q[2]-q[3])*a3*l2*m3
    #     c42 = -qd[1]*a3*l1*m3*sin(q[1]-q[3])
    #     c43 = -qd[2]*a3*l2*m3*sin(q[2]*q[3])
    #     C = np.array([[c11,c12,c13,c14],
    #                   [c21,c22,c23,c24],
    #                   [c31,c32,c33,c34],
    #                   [c41,c42,c43,c44]])
    #
    #     d11 = d0
    #     d12 = d13 = d14 = d21 = d31 = d41 = d24 = d42 = 0
    #     d22 = d1 + d2
    #     d23 = d32 = -d2
    #     d33 = d2 + d3
    #     d34 = d43 = -d3
    #     d44 = d3
    #     D = np.array([[d11, d12, d13, d14],
    #                   [d21, d22, d23, d24],
    #                   [d31, d32, d33, d34],
    #                   [d41, d42, d43, d44]])
    #
    #     g11 = 0
    #     g21 = -g*(a1*m1 + l1*m2 + l1*m3)*sin(q[1])
    #     g31 = -g*(a2*m2 + l2*m3)*sin(q[2])
    #     g41 = -a3*g*m3*sin(q[3])
    #     G = np.array([g11,g21,g31,g41]).T
    #
    #     Minv = np.linalg.inv(M)
    #     tau = np.array([f,0,0,0]).T
    #     qdd = -Minv @ (C @ qd.T + D @ qd.T + G - tau)
    #     xd = np.concatenate([qd,qdd])
    #     return xd

    # def _v2f(self, v):
    #     xdot = deepcopy(self.x[-4])
    #     k = 0.053
    #     R = 1.3
    #     r = 0.005
    #     b = 0.00035
    #     w = xdot/r
    #     I = np.clip(np.array([(v-k*w)/R]),-10.,10.)[0]
    #     torque = k*I - b*w
    #     F = torque/r
    #     return F

    def render(self, mouse_control=False):
        if self.viewer is None:
            self._viewer_setup()
        if not self.viewer.is_alive:
            self._viewer_reset()
        if not mouse_control:
            qpos = [self.x[0], self.x[1]-np.pi, self.x[2] - self.x[1], self.x[3] - self.x[2]]
            qvel = [self.x[4], self.x[5], self.x[6]-self.x[5], self.x[7]-self.x[6]]
            self.data.qpos = qpos
            self.data.qvel = qvel
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def _viewer_setup(self):
        self.model = mujoco.MjModel.from_xml_path('./environment/mujoco/assets/tip.xml')
        self.data = mujoco.MjData(self.model)
        self._viewer_reset()

    def _viewer_reset(self):
        self.viewer = Viewer(model=self.model, data=self.data,
                             width=1000, height=600,
                             title='TripleInvertedPendulumSwing',
                             hide_menus=True)
        self.viewer.cam.distance = self.model.stat.extent * 1.8
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
        fig1, fig_viewport1 = self._set_figure(0, self.viewport.height, 1)
        fig2, fig_viewport2 = self._set_figure(0, 2*int(self.viewport.height/4), 3)
        fig3, fig_viewport3 = self._set_figure(0, 3*int(self.viewport.height/4), 1)
        fig4, fig_viewport4 = self._set_figure(0, 0, 3)
        self.fig = [fig1, fig2, fig3, fig4]
        self.fig_viewport = [fig_viewport1, fig_viewport2, fig_viewport3, fig_viewport4]
        fig1.range[1][0] = -0.5
        fig1.range[1][1] = 0.5
        fig1.linergb = [1, 1, 0]
        fig1.flg_ticklabel[0] = 0
        fig1.linename[0] = 'Pos.'
        fig2.range[1][0] = -4.0
        fig2.range[1][1] = 4.0
        fig2.linename[0] = 'Ang.'
        fig2.flg_ticklabel[0] = 0
        fig3.range[1][0] = -3.
        fig3.range[1][1] = 3.
        fig3.linergb = [1, 1, 0]
        fig3.linename[0] = 'Vel.'
        fig3.flg_ticklabel[0] = 0
        fig4.range[1][0] = -30.
        fig4.range[1][1] = 30.
        fig4.linename[0] = 'Ang.Vel.'
        fig4.flg_ticklabel[0] = 0

    def _set_figure(self, loc_x, loc_y, n):
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        for j in range(n):
            for i in range(0, self.n_timestep):
                fig.linedata[j][2 * i] = float(-i)
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
        sensor_data = (self.data.qpos + np.array([0.,np.pi,0.0,0.0]))
        sensor_data = np.concatenate([sensor_data, self.data.qvel])
        n = [1, 3, 1, 3]
        idx = [[0],[1,2,3],[4],[5,6,7]]
        for i in range(len(self.fig)):
            pnt = int(mujoco.mju_min(self.n_timestep, self.fig[i].linepnt[0] + 1))
            for k in range(n[i]):
                for j in range(pnt-1, 0, -1):
                    self.fig[i].linedata[k][2 * j + 1] = self.fig[i].linedata[k][2 * j - 1]
                self.fig[i].linepnt[k] = pnt
                self.fig[i].linedata[k][1] = sensor_data[idx[i][k]]

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