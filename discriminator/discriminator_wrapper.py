import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import layer_utils
from discriminator.discriminator import CaptionInput, ImageInput, MetadataInput, LstmScalarRewardStrategy, \
    DiscriminatorMaxReward, VisualAttention
from discriminator.discriminator_attend_text import DiscriminatorClassification, TextualAttention, VanillaDotProduct
from discriminator.discriminator_data_utils import create_demo_sampled_batcher
from discriminator.mini_batcher import MiniBatcher, MixedMiniBatcher
from image_utils import image_from_url, visualize_attention
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_losses(train_loss, val_loss, title=""):
    plt.plot(np.arange(0, len(train_loss)), train_loss, 'g', val_loss, 'r')
    train_loss_patch = mpatches.Patch(color='g', label='train')
    val_loss_patch = mpatches.Patch(color='r', label='val')
    plt.legend(handles=[train_loss_patch, val_loss_patch])
    plt.ylabel('Loss')
    plt.xlabel('Iter')
    plt.title(title)
    plt.show()


class DiscriminatorWrapper(object):
    visual_attn_type_key = "visual_attn"
    textual_attn_type_key = "textual_attn"
    vanilla_classification_type_key = "vanilla_cls"

    def __init__(self, train_data, val_data, vocab_data,
                 learner_type=None,
                 hidden_dim=512,
                 load_session=None, saved_model_name=None, model_base_dir="models/discr"):

        self.model_base_dir = model_base_dir
        self.train_data = train_data
        self.val_data = val_data
        self.vocab_data = vocab_data

        self.learner_type = learner_type
        self.demo_batcher, self.sampled_batcher = create_demo_sampled_batcher(self.train_data)
        self.val_demo_batcher, self.val_sample_batcher = create_demo_sampled_batcher(self.val_data)
        self.hidden_dim = hidden_dim

        # Build graph
        if saved_model_name is not None:
            saver = tf.train.import_meta_graph(
                '{}/{}.meta'.format(model_base_dir, saved_model_name))
            saver.restore(load_session, '{}/{}'.format(model_base_dir, saved_model_name))
            graph = load_session.graph
        else:
            graph = None

        caption_input = CaptionInput(word_embedding_init=vocab_data.embedding(), null_id=vocab_data.NULL_ID,
                                     graph=graph)
        image_input = ImageInput(image_feature_dim=train_data.image_features.shape[1:], graph=graph)
        metadata_input = MetadataInput(graph=graph)
        reward_config = self.default_reward_config()

        self.discr, self.learner = self.build_by_learner_type(caption_input,
                                                              image_input,
                                                              metadata_input,
                                                              hidden_dim,
                                                              reward_config,
                                                              graph)

    @staticmethod
    def default_reward_config():
        return LstmScalarRewardStrategy.RewardConfig(
            reward_scalar_transformer=lambda x: tf.nn.tanh(
                layer_utils.affine_transform(x, 1, 'hidden_to_reward'))
        )

    def build_by_learner_type(self, caption_input, image_input, metadata_input, hidden_dim, reward_config, graph):

        print("Building learner {}...".format(self.learner_type))

        if self.learner_type == DiscriminatorWrapper.textual_attn_type_key:

            learner = TextualAttention(self.train_data.max_caption_len,
                                       self.train_data.image_part_num,
                                       self.train_data.image_feature_dim,
                                       self.hidden_dim,
                                       self.hidden_dim)

            discr = DiscriminatorClassification(caption_input, image_input, metadata_input,
                                                is_attention=True,
                                                is_classification=True,
                                                learner_model=learner,
                                                reward_config=reward_config,
                                                hidden_dim=hidden_dim, graph=graph)
        elif self.learner_type == DiscriminatorWrapper.vanilla_classification_type_key:

            learner = VanillaDotProduct(self.hidden_dim)
            discr = DiscriminatorClassification(caption_input, image_input, metadata_input,
                                                is_attention=False,
                                                is_classification=True,
                                                learner_model=learner,
                                                reward_config=reward_config,
                                                hidden_dim=hidden_dim, graph=graph)

        elif self.learner_type == DiscriminatorWrapper.visual_attn_type_key:

            learner = VisualAttention(self.train_data.max_caption_len,
                                      self.train_data.image_part_num,
                                      self.train_data.image_feature_dim,
                                      self.hidden_dim,
                                      self.hidden_dim)

            discr = DiscriminatorMaxReward(caption_input, image_input, metadata_input,
                                           is_attention=True,
                                           learner_model=learner,
                                           reward_config=reward_config,
                                           hidden_dim=hidden_dim, graph=graph)
        else:

            discr = DiscriminatorMaxReward(caption_input, image_input, metadata_input,
                                           is_attention=False,
                                           learner_model=None,
                                           reward_config=reward_config,
                                           hidden_dim=hidden_dim, graph=graph)

        return discr, learner

    def has_attention_model(self):
        return self.train_data.image_part_num is not None

    def pre_train(self, sess, iter_num=400, batch_size=1000, validate=True):

        return self.train(sess,
                          self.demo_batcher,
                          self.sampled_batcher, iter_num, batch_size, validate)

    def train(self, sess, demo_batcher, sampled_batcher, iter_num, batch_size, validate=True):
        train_losses = []
        val_losses = []

        batches = self.batches(demo_batcher, sampled_batcher, batch_size)
        for i, b in enumerate(batches):
            if i >= iter_num:
                break

            image_idx_batch, caption_batch, demo_or_sampled_batch = b

            output = self._train_one_iter(sess, image_idx_batch, caption_batch, demo_or_sampled_batch)

            train_losses.append(output.loss)
            if i % 5== 0:
                print("iter {}, loss: {}".format(i, output.loss))
                if self.discr.is_classification:
                    self.print_classification_result(demo_or_sampled_batch, output.pred, "train")

            if validate:
                if i % 5 == 0:
                    val_loss = self.examine_validation(sess, batch_size, to_examine=False)
                    val_losses.append(val_loss)
                else:
                    val_losses.append(val_losses[-1])

        return train_losses, val_losses

    @staticmethod
    def print_classification_result(demo_or_sampled_batch, pred, split):
        print("{} accuracy: {}".format(split, accuracy_score(y_true=demo_or_sampled_batch, y_pred=pred)))
        print(confusion_matrix(y_true=demo_or_sampled_batch, y_pred=pred))

    def _train_one_iter(self, sess, image_idx_batch, caption_batch, demo_or_sampled_batch):
        caption_batch = caption_batch[:, 1::]
        image_feats_batch = self.train_data.get_image_features(image_idx_batch)
        self.discr.caption_input.pre_feed(caption_word_ids=caption_batch)
        self.discr.image_input.pre_feed(image_features=image_feats_batch)
        self.discr.metadata_input.pre_feed(demo_or_sampled_batch)
        return self.discr.train(sess)

    def _preprocess_online_train_caption(self, caption):
        tokenized = [self.vocab_data.NULL_TOKEN] * self.train_data.max_caption_len
        tokenized[0] = self.vocab_data.START_TOKEN
        for i, tk in enumerate(caption.split()):
            pos = i + 1
            if pos >= self.train_data.max_caption_len:
                break
            tokenized[pos] = tk
        return tokenized

    def online_train(self, sess, iter_num, img_idxs, caption_sentences):

        captions = [self._preprocess_online_train_caption(c) for c in caption_sentences]
        caption_word_idx = self.vocab_data.encode_captions(captions)
        given_size = len(captions)

        # zero label indicating sampled
        new_dat = (img_idxs, caption_word_idx, np.zeros(given_size))
        new_batcher = MiniBatcher(new_dat)
        mixed_sampled_batcher = MixedMiniBatcher([new_batcher, self.sampled_batcher], [0.25, 0.75])
        batch_size = given_size

        return self.train(sess, self.demo_batcher, mixed_sampled_batcher, iter_num, batch_size)

    def process_mini_batch(self, batcher1, batcher2, batch_size):
        return self.merge_image_caption_demo(batcher1.sample(batch_size), batcher2.sample(batch_size))

    def batches(self, batcher1, batcher2, batch_size):
        for b1, b2 in zip(batcher1.batches(batch_size), batcher2.batches(batch_size)):
            yield self.merge_image_caption_demo(b1, b2)

    @staticmethod
    def merge_image_caption_demo(b1, b2):
        image_idx_batch1, caption_batch1, demo_or_sampled_batch1 = b1
        image_idx_batch2, caption_batch2, demo_or_sampled_batch2 = b2
        image_idx_batch = np.concatenate([image_idx_batch1, image_idx_batch2], axis=0)
        caption_batch = np.concatenate([caption_batch1, caption_batch2], axis=0)
        demo_or_sampled_batch = np.concatenate([demo_or_sampled_batch1, demo_or_sampled_batch2], axis=0)
        return image_idx_batch, caption_batch, demo_or_sampled_batch

    def assign_reward(self, sess, img_idxs, caption_sentences,
                      image_idx_from_training=True,
                      to_examine=False,
                      max_step=16):

        captions = [c.split() for c in caption_sentences]

        if image_idx_from_training:
            coco_data = self.train_data
        else:
            coco_data = self.val_data
        image_feats_test = coco_data.get_image_features(img_idxs)
        caption_test = self.vocab_data.encode_captions(captions)

        if self.learner_type == DiscriminatorWrapper.visual_attn_type_key:
            # max len - 1, without start token during train
            assert caption_test.shape[1] < (
                coco_data.max_caption_len - 1), "Attention requires max caption length of {}".format(
                coco_data.max_caption_len - 1)
            additional = coco_data.max_caption_len - caption_test.shape[1] - 1
            nulls = self.vocab_data.get_null_ids((caption_test.shape[0], additional))
            caption_test = np.concatenate((caption_test, nulls), axis=1)

        output = self.run_test(sess, image_feats_test, caption_test)

        if to_examine:
            self.examine(coco_data, img_idxs, caption_test, output, np.ones(image_feats_test.shape[0]))

        if max_step < caption_test.shape[0]:
            rewards = output.masked_reward[:, :max_step]
        else:
            rewards = output.masked_reward

        return output.loss, rewards, output.mean_reward_per_sentence

    def run_validation(self, sess, img_idxs, caption_word_idx, demo_or_sampled_batch):
        image_feats = self.val_data.get_image_features(img_idxs)
        self.discr.image_input.pre_feed(image_feats)
        self.discr.caption_input.pre_feed(caption_word_idx)
        self.discr.metadata_input.pre_feed(labels=demo_or_sampled_batch)
        return self.discr.test(sess)

    def run_test(self, sess, img_feature_test, caption_test, labels=None):
        self.discr.image_input.pre_feed(img_feature_test)
        self.discr.caption_input.pre_feed(caption_test)

        if labels is None:
            labels = np.ones(img_feature_test.shape[0])
        self.discr.metadata_input.pre_feed(labels=labels)
        return self.discr.test(sess)

    def _base_examine(self, chosen_img, chosen_caption, output, examine_func):
        irange = range(len(chosen_img))
        chosen_reward_per_token = output.masked_reward
        chosen_mean_reward = output.mean_reward_per_sentence

        for zipped in zip(irange, chosen_img, chosen_caption, chosen_reward_per_token, chosen_mean_reward):
            examine_func(*zipped)

    def examine(self, coco_data, chosen_img, chosen_caption, output, demo_or_sampled):

        is_classification = self.discr.is_classification

        def two_sf(r):
            return '%s' % float('%.2g' % r)

        def text_mean_reward(i, is_demo, mean_reward):
            if is_classification:
                return "P/G: {}/{} | Avg reward: {}".format(output.pred[i], is_demo, mean_reward)
            else:
                return "G: {} | Avg reward: {}".format(is_demo, mean_reward)

        def examine_visual_attention(i, img_idx, cap, reward, mean_reward):

            is_demo = demo_or_sampled[i]
            decoded = self.vocab_data.decode_captions(cap).split()

            labels = []
            for (c, r) in zip(decoded, reward):
                label = "{}: {}".format(c, two_sf(r))
                labels.append(label)

            print(text_mean_reward(i, is_demo, mean_reward))
            visualize_attention(coco_data.image_paths[img_idx], output.attention[i], labels)
            print("- - - -")

        def examine_reward_by_attentional_word(i, img_idx, cap, reward, mean_reward):
            is_demo = demo_or_sampled[i]
            print(text_mean_reward(i, is_demo, mean_reward))
            self.show_image_by_image_idxs(coco_data, [img_idx])
            decoded = self.vocab_data.decode_captions(cap).split()
            for (j, k, l) in zip(decoded, reward, output.attention[i]):
                print("{:<15} {:<15}, attn: {}".format(j, two_sf(k), two_sf(l)))
            print("- - - -")

        def examine_reward_by_word(i, img_idx, cap, reward, mean_reward):
            is_demo = demo_or_sampled[i]
            print(text_mean_reward(i, is_demo, mean_reward))
            self.show_image_by_image_idxs(coco_data, [img_idx])
            decoded = self.vocab_data.decode_captions(cap).split()
            for (j, k) in zip(decoded, reward):
                print("{:<15} {}".format(j, two_sf(k)))
            print("- - - -")

        if self.learner_type == DiscriminatorWrapper.visual_attn_type_key:
            return self._base_examine(chosen_img, chosen_caption, output, examine_visual_attention)
        elif self.learner_type == DiscriminatorWrapper.textual_attn_type_key:
            return self._base_examine(chosen_img, chosen_caption, output, examine_reward_by_attentional_word)
        else:
            return self._base_examine(chosen_img, chosen_caption, output, examine_reward_by_word)

    def show_image_by_image_idxs(self, coco_data, img_idxs):
        """
            data indices to find image
        """
        urls = coco_data.get_urls_by_image_index(img_idxs)
        for url in urls:
            plt.imshow(image_from_url(url))
            plt.axis('off')
            plt.show()

    def examine_validation(self, sess, batch_size=100, to_examine=True, print_acc=False):
        image_idx_batch, caption_batch, demo_or_sampled_batch = self.process_mini_batch(self.val_demo_batcher,
                                                                                        self.val_sample_batcher,
                                                                                        batch_size)
        caption_batch = caption_batch[:, 1:]
        output = self.run_validation(sess, image_idx_batch, caption_batch, demo_or_sampled_batch)
        if self.discr.is_classification and print_acc:
            self.print_classification_result(demo_or_sampled_batch, output.pred, "val")
        if to_examine:
            self.examine(self.val_data, image_idx_batch, caption_batch, output, demo_or_sampled_batch)
        return output.loss

    def save_model(self, sess, model_name):
        self.discr.save_model(sess, model_name='{}/{}'.format(self.model_base_dir, model_name))
