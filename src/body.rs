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

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct Body {
    pub mass: f32,
    pos: [f32; 3],
    vel: [f32; 3],
}

impl Body {
    pub const fn new_internal(mass: f32, pos: [f32; 3], vel: [f32; 3]) -> Self {
        Body { mass, pos, vel }
    }

    pub fn gravity_force(&self, other: &Body) -> [f32; 3] {
        let mut disp = [0.0; 3];
        let mut dist_sqr = 0.0;
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
            pos: [0.0; 3],
            vel: [0.0; 3],
        };

        body.pos.copy_from_slice(&pos[..3]);
        body.vel.copy_from_slice(&vel[..3]);

        body
    }

    #[wasm_bindgen]
    pub fn get_pos(&self, dim: usize) -> f32 {
        self.pos[dim]
    }

    #[wasm_bindgen]
    pub fn get_vel(&self, dim: usize) -> f32 {
        self.vel[dim]
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
    Rk4,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct BodySystem {
    bodies: Vec<Body>,
}

fn get_forces(bodies: &[Body], force_buf: &mut [[f32; 3]], method: ForceMethod) {
    match method {
        ForceMethod::Naive => get_forces_naive(bodies, force_buf),
    };
}

fn get_forces_naive(bodies: &[Body], force_buf: &mut [[f32; 3]]) {
    let num_bodies = bodies.len();
    for body1_idx in 0..num_bodies {
        for body2_idx in body1_idx + 1..num_bodies {
            let force_comp = bodies[body1_idx].gravity_force(&bodies[body2_idx]);

            for dim_idx in 0..3 {
                force_buf[body1_idx][dim_idx] += force_comp[dim_idx];
                force_buf[body2_idx][dim_idx] -= force_comp[dim_idx];
            }
        }
    }
}

impl BodySystem {
    fn step_bodies_euler(&mut self, timestep: f32, force_method: ForceMethod) {
        let mut force_buf = vec![[0.0; 3]; self.bodies.len()];

        get_forces(&self.bodies[..], &mut force_buf[..], force_method);

        for body_idx in 0..self.bodies.len() {
            for dim_idx in 0..3 {
                self.bodies[body_idx].pos[dim_idx] += timestep * self.bodies[body_idx].vel[dim_idx];
            }

            for dim_idx in 0..3 {
                self.bodies[body_idx].vel[dim_idx] +=
                    timestep * force_buf[body_idx][dim_idx] / self.bodies[body_idx].mass;
            }
        }
    }

    fn step_bodies_rk4(&mut self, timestep: f32, force_method: ForceMethod) {
        let mut force_buf: [Vec<[f32; 3]>; 4] =
            core::array::from_fn(|_| vec![[0.0; 3]; self.bodies.len()]);
        let mut body_buf: [Vec<Body>; 3] = core::array::from_fn(|_| self.bodies.clone());

        get_forces(&self.bodies[..], &mut force_buf[0][..], force_method);

        for body_idx in 0..self.bodies.len() {
            for dim_idx in 0..3 {
                body_buf[0][body_idx].pos[dim_idx] +=
                    timestep / 2.0 * self.bodies[body_idx].vel[dim_idx];
            }

            for dim_idx in 0..3 {
                body_buf[0][body_idx].vel[dim_idx] +=
                    timestep / 2.0 * force_buf[0][body_idx][dim_idx] / self.bodies[body_idx].mass;
            }
        }

        get_forces(&body_buf[0][..], &mut force_buf[1][..], force_method);

        for body_idx in 0..self.bodies.len() {
            for dim_idx in 0..3 {
                body_buf[1][body_idx].pos[dim_idx] +=
                    timestep / 2.0 * body_buf[0][body_idx].vel[dim_idx];
            }

            for dim_idx in 0..3 {
                body_buf[1][body_idx].vel[dim_idx] +=
                    timestep / 2.0 * force_buf[1][body_idx][dim_idx] / self.bodies[body_idx].mass;
            }
        }

        get_forces(&body_buf[1][..], &mut force_buf[2][..], force_method);

        for body_idx in 0..self.bodies.len() {
            for dim_idx in 0..3 {
                body_buf[2][body_idx].pos[dim_idx] += timestep * body_buf[1][body_idx].vel[dim_idx];
            }

            for dim_idx in 0..3 {
                body_buf[2][body_idx].vel[dim_idx] +=
                    timestep * force_buf[2][body_idx][dim_idx] / self.bodies[body_idx].mass;
            }
        }

        get_forces(&body_buf[2][..], &mut force_buf[3][..], force_method);

        for body_idx in 0..self.bodies.len() {
            for dim_idx in 0..3 {
                self.bodies[body_idx].pos[dim_idx] += timestep / 6.0
                    * (self.bodies[body_idx].vel[dim_idx]
                        + 2.0 * body_buf[0][body_idx].vel[dim_idx]
                        + 2.0 * body_buf[1][body_idx].vel[dim_idx]
                        + body_buf[2][body_idx].vel[dim_idx]);
            }

            for dim_idx in 0..3 {
                self.bodies[body_idx].vel[dim_idx] += timestep / 6.0
                    * (force_buf[0][body_idx][dim_idx]
                        + 2.0 * force_buf[1][body_idx][dim_idx]
                        + 2.0 * force_buf[2][body_idx][dim_idx]
                        + force_buf[3][body_idx][dim_idx])
                    / self.bodies[body_idx].mass;
            }
        }
    }
}

#[wasm_bindgen]
impl BodySystem {
    #[wasm_bindgen]
    pub fn new(bodies: Vec<Body>) -> Self {
        BodySystem { bodies }
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
            StepMethod::Rk4 => self.step_bodies_rk4(timestep, force_method),
        };
    }

    #[wasm_bindgen]
    pub fn get_bodies(&self) -> Vec<Body> {
        self.bodies.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn test_force_calculation() {
        let body1 = Body::new_internal(1.0, [0.0; 3], [0.0; 3]);
        let body2 = Body::new_internal(2.0, [5.0, 0.0, 0.0], [0.0; 3]);
        let force = body1.gravity_force(&body2);

        const TARGET_FORCE: [f32; 3] = [0.08, 0.0, 0.0];

        for i in 0..3 {
            assert!(
                (force[i] - TARGET_FORCE[i]).abs() < EPS,
                "Incorrect force in dimension {}",
                i + 1
            );
        }
    }

    struct StepTestInput {
        bodies: Vec<Body>,
        timestep: f32,
        force_method: ForceMethod,
        step_method: StepMethod,
    }

    struct StepTestOutput<'a> {
        bodies: &'a [Body],
    }

    struct StepTestData<'a> {
        input: StepTestInput,
        output: StepTestOutput<'a>,
    }

    macro_rules! add_step_testcase {
        ($testname:ident, $value:expr) => {
            #[test]
            fn $testname() {
                let test_data: StepTestData = $value;

                assert_eq!(
                    test_data.input.bodies.len(),
                    test_data.output.bodies.len(),
                    "Invalid test data; expected same number of bodies"
                );

                for body_idx in 0..test_data.output.bodies.len() {
                    assert_eq!(
                        test_data.input.bodies[body_idx].mass,
                        test_data.output.bodies[body_idx].mass,
                        "Invalid test data; expected same mass for input and output"
                    );
                }

                let mut body_system = BodySystem::new(test_data.input.bodies);

                body_system.step_bodies(
                    test_data.input.timestep,
                    test_data.input.force_method,
                    test_data.input.step_method,
                );

                let result_bodies = body_system.get_bodies();
                assert_eq!(
                    result_bodies.len(),
                    test_data.output.bodies.len(),
                    "Did not get same number of bodies in result"
                );

                for body_idx in 0..test_data.output.bodies.len() {
                    assert_eq!(
                        result_bodies[body_idx].mass, test_data.output.bodies[body_idx].mass,
                        "Different mass in result bodies"
                    );
                    for dim_idx in 0..3 {
                        assert!(
                            (result_bodies[body_idx].pos[dim_idx]
                                - test_data.output.bodies[body_idx].pos[dim_idx])
                                .abs()
                                < EPS,
                            "Incorrect position for body {}
   result: {:?}
 expected: {:?}",
                            body_idx + 1,
                            result_bodies[body_idx],
                            test_data.output.bodies[body_idx]
                        );
                        assert!(
                            (result_bodies[body_idx].vel[dim_idx]
                                - test_data.output.bodies[body_idx].vel[dim_idx])
                                .abs()
                                < EPS,
                            "Incorrect velocity for body {}
   result: {:?}
 expected: {:?}",
                            body_idx + 1,
                            result_bodies[body_idx],
                            test_data.output.bodies[body_idx],
                        );
                    }
                }
            }
        };
    }

    add_step_testcase!(
        test_euler_step,
        StepTestData {
            input: StepTestInput {
                bodies: vec![
                    Body::new_internal(2.0, [0.0; 3], [-1.0, 0.0, 0.0]),
                    Body::new_internal(3.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
                ],
                timestep: 0.01,
                force_method: ForceMethod::Naive,
                step_method: StepMethod::Euler,
            },
            output: StepTestOutput {
                bodies: &[
                    Body::new_internal(2.0, [-0.01, 0.0, 0.0], [-0.97, 0.0, 0.0]),
                    Body::new_internal(3.0, [1.01, 0.0, 0.0], [0.98, 0.0, 0.0]),
                ],
            },
        }
    );

    add_step_testcase!(
        test_euler_step_2,
        StepTestData {
            input: StepTestInput {
                bodies: vec![
                    Body::new_internal(5.54, [-0.3, -0.16, 0.06], [0.62, 0.85, -0.13]),
                    Body::new_internal(6.34, [0.3, -0.94, -0.28], [-0.29, -0.25, 0.08]),
                    Body::new_internal(6.6, [-0.34, -0.81, 0.91], [-0.18, 0.82, -0.27])
                ],
                timestep: 0.01,
                force_method: ForceMethod::Naive,
                step_method: StepMethod::Euler,
            },
            output: StepTestOutput {
                bodies: &[
                    Body::new_internal(
                        5.54,
                        [-0.2938, -0.1515, 0.0587],
                        [0.651554939425, 0.77124194633, -0.103407095465]
                    ),
                    Body::new_internal(
                        6.34,
                        [0.2971, -0.9425, -0.2792],
                        [-0.336340120551, -0.208281829426, 0.128090614184]
                    ),
                    Body::new_internal(
                        6.6,
                        [-0.3418, -0.8018, 0.9073],
                        [-0.161972424261, 0.846034305438, -0.33851805834]
                    )
                ],
            },
        }
    );

    add_step_testcase!(
        test_rk4_step,
        StepTestData {
            input: StepTestInput {
                bodies: vec![
                    Body::new_internal(2.0, [0.0; 3], [-1.0, 0.0, 0.0]),
                    Body::new_internal(3.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
                ],
                timestep: 0.01,
                force_method: ForceMethod::Naive,
                step_method: StepMethod::Rk4,
            },
            output: StepTestOutput {
                bodies: &[
                    Body::new_internal(
                        2.0,
                        [-0.00985195826043, 0.0, 0.0],
                        [-0.97058349796, 0.0, 0.0]
                    ),
                    Body::new_internal(3.0, [1.00990130551, 0.0, 0.0], [0.98038899864, 0.0, 0.0]),
                ],
            },
        }
    );
}
