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

    //let mut simple_model =
    Cpu::default().build_module::<dfdx::nn::builders::Linear<1, { Game::ACTION_SIZE }>, f32>();

    let adam_config = AdamConfig {
        lr: 1e-4,
        //weight_decay: Some(WeightDecay::L2(1e-3)),
        //betas: [0.9, 0.999],
        ..Default::default()
    };
    let mut optimizer = Adam::new(&model.0, adam_config);

    let temp = 0.5;
    //let mut grads = simple_model.alloc_grads();

    let mut rng = rand::prelude::StdRng::from_seed([42; 32]);
    for i in 0..2_i32.pow(12) {
        let (batch, actions) = Game::get_batch::<16, _>(&mut rng, None);

        // println!("S: {:?}", batch.clone().0.array());
        // println!("A: {:?}", batch.clone().1.array());
        // println!("R: {:?}", batch.clone().2.array());
        // println!("T: {:?}", batch.clone().3.array());

        let mut prev_loss = f32::MAX;
        for epoch in 0..1 {
            let b = batch.clone();
            let loss = model.train_on_batch(b, actions, &mut optimizer);

            // let rtg = batch
            //     .2
            //     .clone()
            //     .select(Cpu::default().tensor([{ TestConfig::SEQ_LEN - 1 }; 64]));
            // println!("{:?}", rtg.array());
            // let out = simple_model.forward_mut(rtg.traced(grads));
            // println!("{:?}", out.array());
            // let actual = actions
            //     .map(|action| Game::action_to_tensor(&action))
            //     .stack();
            // println!("{:?}", actual.array());

            // let g_loss = bce_with_logits(out, actual).mean();
            // let loss = g_loss.as_vec()[0];
            // grads = g_loss.backward();

            // optimizer.update(&mut simple_model, &grads).unwrap();
            // simple_model.zero_grads(&mut grads);
            println!("Loss at batch {i} epoch {epoch}: {loss:.3} (offline learn)\r");

            //Early stopping
            if (loss - prev_loss).abs() < 1e-3 {
                break;
            } else {
                prev_loss = loss;
            }
        }

        if (i - 1) % 128 == 0 {
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
