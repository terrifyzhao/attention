from annlp.trainer_v2.trainer import TMTextTrainer
import pandas as pd
from transformers import AdamW
import torch
from model import Model
from annlp import print_sentence_length


class MyTrainer(TMTextTrainer):
    def get_train_data(self):
        train = pd.read_csv('tnews_public/train.csv')

        print(train['label'].value_counts())
        train_text = train['text'].tolist()
        train_label = train['label'].tolist()
        print_sentence_length(train_text)
        return self.tokenize(train_text), train_label

    def get_dev_data(self):
        dev = pd.read_csv('tnews_public/dev.csv')
        dev_text = dev['text'].tolist()
        dev_label = dev['label'].astype('int')

        return self.tokenize(dev_text), dev_label

    def configure_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def train_step(self, data):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        labels = data['labels'].to(self.device).long()

        out = self.model(input_ids)
        out = torch.mean(out, dim=1)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(out, labels)

        result = {'loss': loss,
                  'pred': torch.argmax(out, dim=-1).cpu().numpy(),
                  'label': labels.cpu().numpy()}
        return result


if __name__ == '__main__':

    for i in range(12):
        print('层数：', i + 1)
        model = Model(21128, 768, 4, i + 1)
        trainer = MyTrainer(model=model, max_length=32, batch_size=64, lr=1e-3, monitor='acc',use_amp=True)
        import time

        s = time.time()
        trainer.run()
        print('总耗时:')
        print(time.time() - s)
