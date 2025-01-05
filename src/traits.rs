use decision_transformer_dfdx::{DTModelConfig, DTState, GetOfflineData, HumanEvaluatable};
use dfdx::prelude::*;

use crate::game::{Action, Game};

pub struct TestConfig;

impl DTModelConfig for TestConfig {
    const NUM_ATTENTION_HEADS: usize = 4;
    const HIDDEN_SIZE: usize = 64;
    const MLP_INNER: usize = 4 * 64;
    const SEQ_LEN: usize = 4;
    const MAX_EPISODES_IN_GAME: usize = 6;
    const NUM_LAYERS: usize = 4;
}

impl DTState<f32, Cpu, TestConfig> for Game {
    type Action = Action;

    const STATE_SIZE: usize = 5;

    const ACTION_SIZE: usize = 2;

    fn new_random<R: rand::Rng + ?Sized>(_rng: &mut R) -> Self {
        Self::new()
    }

    fn apply_action(&mut self, action: Self::Action) {
        self.apply(action)
    }

    fn get_reward(&self, action: Self::Action) -> f32 {
        match action {
            Action::Gain => 1.0,
            Action::Die => -0.0,
        }
    }

    fn to_tensor(&self) -> Tensor<(Const<{ Self::STATE_SIZE }>,), f32, Cpu> {
        let mut t: Tensor<(Const<{ Self::STATE_SIZE }>,), f32, Cpu> = Cpu::default().zeros();

        t[[self.points]] = 1.0;

        t
    }

    fn action_to_index(action: &Self::Action) -> usize {
        match action {
            Action::Gain => 0,
            Action::Die => 1,
        }
    }

    fn index_to_action(action: usize) -> Self::Action {
        match action {
            0 => Action::Gain,
            1 => Action::Die,
            _ => unreachable!(),
        }
    }
}

impl HumanEvaluatable<f32, Cpu, TestConfig> for Game {
    fn print(&self) {
        println!("{self:?}");
    }

    fn print_action(action: &Self::Action) {
        println!("{action:?}")
    }

    fn is_still_playing(&self) -> bool {
        self.still_playing
    }
}

impl GetOfflineData<f32, Cpu, TestConfig> for Game {
    fn play_one_game<R: rand::Rng + ?Sized>(rng: &mut R) -> (Vec<Self>, Vec<Self::Action>) {
        let mut states = vec![];
        let mut actions = vec![];

        let mut state = Self::new();

        while state.still_playing {
            states.push(state.clone());

            let action = [Action::Die, Action::Gain][rng.gen_range(0..2)];

            actions.push(action.clone());

            state.apply(action);
        }

        (states, actions)
    }
}
