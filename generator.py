class AudioGenerator(keras.utils.Sequence):
  
  def __init__(self, audio_files, input_size, output_size, gap_size, start_output_pct, end_output_pct, batch_size):
    self.audio_files = audio_files
    self.input_size = input_size
    self.output_size = output_size
    self.gap_size = gap_size
    self.start_output_pct = start_output_pct
    self.end_output_pct = end_output_pct
    self.batch_size = batch_size
    self.audio_length = np.sum(self.audio_files.lengths)
    #TODO: assert all audios are at least as long as input size
    
    
  def __len__(self):
    return (np.ceil(float(self.audio_length - len(self.audio_files)*(self.input_size + self.gap_size)) * (self.end_output_pct - self.start_output_pct) / 100.0 / float(self.batch_size))).astype(np.int)
  
  
  def get_start_idx(self, idx):
    return file_idx, start_idx

  def get_mini_batch(self, file_idx, start_output_idx, mini_batch_size):
    file_name = self.audio_files.names[file_idx]
    audio = tfio.audio.AudioIOTensor('../content/all_audios/' + str(file_name)).to_tensor()
    end_output_idx = min(self.audio_files.lengths[file_idx]-(self.input_size + self.gap_size), start_output_idx + mini_batch_size)
    num_samples = en
    batch_x = np.zeros((num_train_samples, input_length,1))
    batch_y = np.zeros((num_train_samples, output_length,1))

  def __getitem__(self, idx):

    batch_x = self.audio_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            resize(imread('/content/all_audios/' + str(file_name)), (80, 80, 3))
               for file_name in batch_x])/255.0, np.array(batch_y)