#[derive(Debug, Clone, Copy)]
pub struct Game {
    pub points: usize,
    pub still_playing: bool,
}

impl Game {
    pub fn new() -> Self {
        Self {
            points: 0,
            still_playing: true,
        }
    }

    pub fn apply(&mut self, action: Action) {
        match action {
            Action::Gain => {
                self.points += 1;
                if self.points == 4 {
                    self.still_playing = false;
                }
            }
            Action::Die => self.still_playing = false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Gain,
    Die,
}
