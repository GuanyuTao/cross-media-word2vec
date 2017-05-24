# coding=utf8
import word2vec


flags = word2vec.flags

flags.DEFINE_string("nce_vector_path", sys.argv[4], "File name for NCE vector sample source")
flags.DEFINE_string("nce_theta_w_path", sys.argv[5], "File name for NCE theta_w sample source")
flags.DEFINE_string("nce_theta_b_path", sys.argv[6], "File name for NCE theta_b sample source")


class NCE_Word2Vec(Word2Vec):
    """Noise-Contrastive Estimation Word2Vec model (Skipgram)."""

    def read_nce(self):
        opt = self._options
        nce_vector = []
        nce_theta_w = []
        nce_theta_b = []

        if opt.nce_path != '':
            with open(opt.nce_vector_path, 'r') as f_vector,\
                    open(opt.nce_theta_w_path, 'r') as f_theta_w,\
                    open(opt.nce_theta_b_path, 'r') as f_theta_b:
                for _line in f_vector:
                    _content = _line.strip().split('\t')
                    if len(_content) != 2:
                        continue
                    _word, _vector = _content
                    nce_vector.append(map(float, _vector.split(' ')))

                for _line in f_theta_w:
                    nce_theta_w.append(map(float, _line.split(' ')))

                for _line in f_theta_b:
                    nce_theta_b.append(float(_line))
                
                # All Constant (Used as distribution)
                
                # Embedding (trained before at other training set): [vocab_size, emb_dim]
                self._nce_vector = tf.constant(nce_vector)

                # Theta_w (trained before ... ): [vocab_size, emb_dim]
                self._nce_theta_w = tf.constant(nce_theta_w)

                # Theta_b (trained before ... ): [1, vocab_size]
                self._nce_theta_b = tf.constant(nce_theta_b)

        else:
            print('Error: No nce path determined!')                     

    def forward(self, examples, labels):
        """Build the graph for the forward pass(Adding NCE part)."""
        
        opts = self._options

        # Declare all variables we need.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
                tf.random_uniform(
                        [opts.vocab_size, opts.emb_dim], -init_width, init_width),
                name="emb")
        self._emb = emb

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
                tf.zeros([opts.vocab_size, opts.emb_dim]),
                name="sm_w_t")

        # Softmax bias: [vocab_size].
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")
        
        self._theta_w = sm_w_t
        self._theta_b = sm_b

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
                tf.cast(labels,dtype=tf.int64), 
                [opts.batch_size, 1]
                )

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # Negative sampling.
        sampled_ids = self.nce_sample(labels_matrix, example_emb)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])

        sampled_logits = tf.matmul(example_emb,
                sampled_w,
                transpose_b=True) + sampled_b_vec

        return true_logits, sampled_logits



    def build_graph(self):
        """Build the graph for the full model."""
        opts = self._options
        # The training data. A text file.
        (words, counts, words_per_epoch, self._epoch, self._words, examples,
         labels) = word2vec.skipgram(filename=opts.train_data,
                 batch_size=opts.batch_size,
                 window_size=opts.window_size,
                 min_count=opts.min_count,
                 subsample=opts.subsample)
        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        
       
        
        print("Load NCE sample...")
        self.read_nce()
        print("NCE file: ", opts.nce_path)
        


        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i
        true_logits, sampled_logits = self.forward(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        # scalar_summary("NCE loss", loss)
        self._loss = loss
        self.optimize(loss)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        # tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(
                {
                    "emb": self._emb
                    }
                )

  
    def nce_sample(self, labels, batch_emb):
        """Return NCE sample"""
        opts = self._options

        # media_matrix=tf.matmul(batch_emb,self._nce_theta_w,transpose_b=True)+self._nce_theta_b  
        
        # nce_distribution = tf.reduce_sum(media_matrix, 0) / self._options.batch_size
        
        unigrams = [1 for _ in xrange(opts.vocab_size)]

        rough_sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels,
                num_true=1,
                num_sampled=opts.num_samples * 5, # in order to find top k
                unique=True,
                range_max=opts.vocab_size,
                distortion=0.75,
                unigrams=unigrams))

        real_nce_theta_w = tf.nn.embedding_lookup(self._nce_theta_w, rough_sampled_ids)
        real_nce_theta_b = tf.nn.embedding_lookup(self._nce_theta_b, rough_sampled_ids)

        media_matrix = tf.matmul(batch_emb, real_nce_theta_w, transpose_b=True) + real_nce_theta_b

        rough_sampled_logits = tf.reduce_sum(media_matrix, 0) / self._options.batch_size

        # rough_sampled_logits = tf.nn.embedding_lookup(nce_distribution, rough_sampled_ids)

        _, top_k_sampled_logits_id = tf.nn.top_k(rough_sampled_logits, opts.num_samples)

        sampled_ids = tf.nn.embedding_lookup(rough_sampled_ids, top_k_sampled_logits_id)

        return sampled_ids


