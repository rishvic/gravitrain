/* Copyright 2024 Rishvic Pushpakaran
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.  */

use wasm_bindgen::prelude::*;

const BUFFER_DEPTH: usize = 4;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct Body {
    mass: f32,
    pos: [f32; 3],
    vel: [f32; 3],
}

impl Body {
    pub const fn new_internal(mass: f32, pos: [f32; 3], vel: [f32; 3]) -> Self {
        Body { mass, pos, vel }
    }

    pub fn gravity_force(&self, other: &Body) -> [f32; 3] {
        let mut disp = [0f32; 3];
        let mut dist_sqr = 0f32;
        for dim_idx in 0..3 {
            disp[dim_idx] = other.pos[dim_idx] - self.pos[dim_idx];
            dist_sqr += disp[dim_idx] * disp[dim_idx];
        }

        let force_factor = self.mass * other.mass / dist_sqr.powf(1.5);
        for dim_idx in 0..3 {
            disp[dim_idx] *= force_factor;
        }

        disp
    }
}

#[wasm_bindgen]
impl Body {
    #[wasm_bindgen]
    pub fn new(mass: f32, pos: Vec<f32>, vel: Vec<f32>) -> Self {
        let mut body = Body {
            mass,
            pos: [0f32; 3],
            vel: [0f32; 3],
        };

        body.pos.copy_from_slice(&pos[..3]);
        body.vel.copy_from_slice(&vel[..3]);

        body
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ForceMethod {
    Naive,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum StepMethod {
    Euler,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct BodySystem {
    bodies: [Vec<Body>; BUFFER_DEPTH],
    forces: [Vec<[f32; 3]>; BUFFER_DEPTH],
}

impl BodySystem {
    fn step_bodies_euler(&mut self, timestep: f32, force_method: ForceMethod) {
        self.forces[0].fill([0f32; 3]);
        self.get_forces(0, force_method);

        for body_idx in 0..self.bodies[0].len() {
            for dim_idx in 0..3 {
                self.bodies[0][body_idx].pos[dim_idx] +=
                    timestep * self.bodies[0][body_idx].vel[dim_idx];
            }

            for dim_idx in 0..3 {
                self.bodies[0][body_idx].vel[dim_idx] +=
                    timestep * self.forces[0][body_idx][dim_idx];
            }
        }
    }

    fn get_forces(&mut self, buf_ind: usize, method: ForceMethod) {
        match method {
            ForceMethod::Naive => self.get_forces_naive(buf_ind),
        };
    }

    fn get_forces_naive(&mut self, buf_ind: usize) {
        let num_bodies = self.bodies[0].len();
        for body1_idx in 0..num_bodies {
            for body2_idx in body1_idx + 1..num_bodies {
                let force_comp =
                    self.bodies[buf_ind][body1_idx].gravity_force(&self.bodies[buf_ind][body2_idx]);

                for dim_idx in 0..3 {
                    self.forces[buf_ind][body1_idx][dim_idx] += force_comp[dim_idx];
                    self.forces[buf_ind][body2_idx][dim_idx] -= force_comp[dim_idx];
                }
            }
        }
    }
}

#[wasm_bindgen]
impl BodySystem {
    #[wasm_bindgen]
    pub fn new(bodies: Vec<Body>) -> Self {
        let num_bodies = bodies.len();

        let mut body_system = BodySystem {
            bodies: std::array::from_fn(|_| {
                vec![Body::new_internal(0f32, [0f32; 3], [0f32; 3]); num_bodies]
            }),
            forces: std::array::from_fn(|_| vec![[0f32; 3]; num_bodies]),
        };
        body_system.bodies[0] = bodies;

        body_system
    }

    #[wasm_bindgen]
    pub fn step_bodies(
        &mut self,
        timestep: f32,
        force_method: ForceMethod,
        step_method: StepMethod,
    ) {
        match step_method {
            StepMethod::Euler => self.step_bodies_euler(timestep, force_method),
        };
    }

    #[wasm_bindgen]
    pub fn get_bodies(&self) -> Vec<Body> {
        self.bodies[0].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6f32;

    #[test]
    fn test_force_calculation() {
        let body1 = Body::new_internal(1f32, [0f32; 3], [0f32; 3]);
        let body2 = Body::new_internal(2f32, [5f32, 0f32, 0f32], [0f32; 3]);
        let force = body1.gravity_force(&body2);

        const TARGET_FORCE: [f32; 3] = [0.08f32, 0f32, 0f32];

        for i in 0..3 {
            assert!(
                (force[i] - TARGET_FORCE[i]).abs() < EPS,
                "Incorrect force in dimension {}",
                i + 1
            );
        }
    }

    #[test]
    fn test_euler_step() {
        let body1 = Body::new_internal(1f32, [0f32; 3], [-1f32, 0f32, 0f32]);
        let body2 = Body::new_internal(1f32, [1f32, 0f32, 0f32], [1f32, 0f32, 0f32]);

        let mut body_system = BodySystem::new(vec![body1, body2]);

        body_system.step_bodies(0.1f32, ForceMethod::Naive, StepMethod::Euler);

        let result_bodies = body_system.get_bodies();

        const TARGET_BODIES: [Body; 2] = [
            Body::new_internal(1f32, [-0.1f32, 0f32, 0f32], [-0.9f32, 0f32, 0f32]),
            Body::new_internal(1f32, [1.1f32, 0f32, 0f32], [0.9f32, 0f32, 0f32]),
        ];

        for body_idx in 0..2 {
            for dim_idx in 0..3 {
                assert!(
                    (result_bodies[body_idx].pos[dim_idx] - TARGET_BODIES[body_idx].pos[dim_idx])
                        .abs()
                        < EPS,
                    "Incorrect position for body {}, dimension {}",
                    body_idx + 1,
                    dim_idx + 1
                );
                assert!(
                    (result_bodies[body_idx].vel[dim_idx] - TARGET_BODIES[body_idx].vel[dim_idx])
                        .abs()
                        < EPS,
                    "Incorrect velocity for body {}, dimension {}",
                    body_idx + 1,
                    dim_idx + 1
                );
            }
        }
    }
}
