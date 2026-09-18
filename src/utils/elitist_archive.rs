use serde::{Deserialize, Serialize};

use crate::components::cgp_components::chromosome::Chromosome;

#[derive(Clone, Serialize, Deserialize)]
pub struct ElitistArchive {
    pub capacity: usize,
    pub elitists: Vec<(Chromosome, f32)>,
}

impl ElitistArchive {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            elitists: Vec::with_capacity(capacity),
        }
    }

    // Try to archive a chromosome
    pub fn try_archive(&mut self, candidate: &Chromosome, fitness: f32) {
        if self.capacity == 0 {
            return;
        }

        let is_already_archived = self.elitists.iter().any(|(c, _)| c.nodes_grid == candidate.nodes_grid);

        if is_already_archived {
            return;
        }

        let worst_archived_fitness = self.elitists.last().map(|(_, f)| *f).unwrap_or(f32::INFINITY);

        if self.elitists.len() < self.capacity || fitness <= worst_archived_fitness {
            // Allows neutral drift by inserting newer individuals of the same fitness to the left of older ones, ensuring they get preferred when truncating
            let insert_pos = self.elitists.iter().position(|(_, f)| fitness <= *f).unwrap_or(self.elitists.len());
            self.elitists.insert(insert_pos, (candidate.clone(), fitness));
            self.elitists.truncate(self.capacity);
        }
    }

    // Returns a sorted vector of chromosomes in the archive (lowest to highest fitness)
    pub fn get_chromosomes(&self) -> Vec<Chromosome> {
        return self.elitists.iter().map(|(c, _)| c.clone()).collect();
    }

    // Returns a sorted vector of fitness values for all chromosomes in the archive (lowest to highest fitness)
    pub fn get_fitness(&self) -> Vec<f32> {
        return self.elitists.iter().map(|(_, f)| *f).collect();
    }

    // Returns (Chromosome, fitness) tuple of best individual in the archive
    pub fn get_best(&self) -> Option<&(Chromosome, f32)> {
        return self.elitists.first();
    }

    // Returns number of chromosomes currently in the archive
    pub fn len(&self) -> usize {
        return self.elitists.len();
    }

    // Returns true if the archive is empty
    pub fn is_empty(&self) -> bool {
        return self.elitists.is_empty();
    }

    // Resets the archive, deleting all currently archived elitists
    pub fn clear_archive(&mut self) {
        self.elitists.clear();
    }
}