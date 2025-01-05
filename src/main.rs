#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod game;
mod traits;

use decision_transformer_dfdx::{DTState, GetOfflineData};
use dfdx::{
    nn::DeviceBuildExt,
    optim::{Adam, AdamConfig},
    tensor::Cpu,
};
use game::Game;
use rand::SeedableRng;

fn main() {
    let mut model = Game::build_model();

    let adam_config = AdamConfig {
        lr: 1e-4,
        //weight_decay: Some(WeightDecay::L2(1e-3)),
        //betas: [0.9, 0.999],
        ..Default::default()
    };
    let mut optimizer = Adam::new(&model.0, adam_config);

    let temp = 0.5;

    let mut rng = rand::prelude::StdRng::from_seed([42; 32]);
    for i in 0..2_i32.pow(10) {
        let (batch, actions) = Game::get_batch::<16, _>(&mut rng, None);

        let mut prev_loss = f32::MAX;
        for epoch in 0..1 {
            let b = batch.clone();
            let loss = model.train_on_batch(b, actions, &mut optimizer);

            println!("Loss at batch {i} epoch {epoch}: {loss:.3} (offline learn)\r");

            //Early stopping
            if (loss - prev_loss).abs() < 1e-3 {
                break;
            } else {
                prev_loss = loss;
            }
        }

        if (i - 1) % 128 == 0 {
            // Evaluate the model, varying the desired_reward parameter
            // This reflects the cumulative reward the decision transformer should seek to achieve
            // In our case, since each "point" is one point of reward, the desired_reward should simply equal the number of times the model should to gain a point

            let s = model.evaluate(Game::new(), temp, 0.0, false);
            println!("Achieved {}/0 points", s.points);

            let s = model.evaluate(Game::new(), temp, 1.0, false);
            println!("Achieved {}/1 points", s.points);

            let s = model.evaluate(Game::new(), temp, 2.0, false);
            println!("Achieved {}/2 points", s.points);

            let s = model.evaluate(Game::new(), temp, 3.0, false);
            println!("Achieved {}/3 points", s.points);

            let s = model.evaluate(Game::new(), temp, 4.0, false);
            println!("Achieved {}/4 points", s.points);
        }
    }
}
