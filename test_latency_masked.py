import torch
import os
import numpy as np
import time
import copy
from models.modeling_llama import LlamaForCausalLM
from models.modeling_llama import LlamaConfig
from models.tokenization_llama import LlamaTokenizer

HEAD_DIM = 4096 // 32

def load_zs(peft):
    if peft == 'llm_pruner':
        hidden_z = torch.from_numpy(np.ones(4096))
        head_z = torch.from_numpy(np.ones((32, 32)))
        intermediate_z = torch.from_numpy(np.ones((32, 11008)))
        for i in range(3,30):
            head_z[i][:int(32*0.6)] = 0
            intermediate_z[i][:int(11008*0.6)] = 0
        head_z = head_z.reshape(32, 1, 32, 1, 1)
        intermediate_z = intermediate_z.reshape(32, 1, 1, 11008)
        return hidden_z, head_z, intermediate_z
    zs = torch.load(os.path.join(peft, 'zs.pt'), map_location="cpu")
    hidden_z = zs['hidden_z'] if 'hidden_z' in zs.keys() else torch.from_numpy(np.ones(4096))
    head_z = zs['head_z']
    intermediate_z = zs['intermediate_z']
    hidden_z = hidden_z.detach()
    head_z = head_z.detach()
    intermediate_z = intermediate_z.detach()
    return hidden_z, head_z, intermediate_z


def load_model(base_model):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    config = LlamaConfig.from_pretrained(
        base_model,
    )
    config.use_lora = False
    model = LlamaForCausalLM.from_pretrained(
        LlamaForCausalLM,
        base_model,
        low_cpu_mem_usage=True,
        config=config,
    )
    return model.half().to('cuda'), tokenizer

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.numel()
    # print(f'param count: {param_count}')

def model_latency(model, tokenizer, size, zs=None):
    hidden_z, head_z, intermediate_z = None, None, None
    if zs is not None:
        hidden_z, head_z, intermediate_z = zs
        hidden_z = hidden_z.to('cuda')
        head_z = head_z.to('cuda')
        intermediate_z = intermediate_z.to('cuda')
    count_param(model)
    assert size in [256, 512, 1024, 2048]

    # if size == 256:
    #     text = 'Select your preferences and run the install command. Stable represents the most currently tested and supported version of PyTorch. This should be suitable for many users. Preview is available if you want the latest, not fully tested and supported, builds that are generated nightly. Please ensure that you have met the prerequisites below (e.g., numpy), depending on your package manager. Anaconda is our recommended package manager since it installs all dependencies. You can also install previous versions of PyTorch. Note that LibTorch is only available for C++.'
    #     input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    #     input_ids = torch.cat([input_ids, torch.zeros((1, 256-input_ids.shape[1]), dtype=torch.long).to('cuda')], dim=1)
    # elif size == 512:
    #     text = 'Ikuhara was a director on the television anime adaptation of Sailor Moon at Toei Animation in the 1990s; after growing frustrated by the lack of creative control in directing an adapted work, he departed the company in 1995 to create an original series. While he initially conceived of Utena as a mainstream shōjo series aimed at capitalizing on the commercial success of Sailor Moon, the direction of the series shifted dramatically during production towards an avant-garde and surrealist tone. The series has been described as a deconstruction and subversion of fairy tales and the magical girl genre of shōjo manga, making heavy use of allegory and symbolism to comment on themes of gender, sexuality, and coming-of-age. Its visual and narrative style is characterized by a sense of theatrical presentation and staging, drawing inspiration from the all-female Japanese theater troupe the Takarazuka Revue, as well as the experimental theater of Shūji Terayama, whose frequent collaborator J. A. Seazer created the songs featured in the series. Revolutionary Girl Utena has been the subject of both domestic and international critical acclaim, and has received many accolades. It has been praised for its treatment of LGBT themes and subject material, and has influenced subsequent animated works. A manga adaptation of Utena written and illustrated Saito was developed contemporaneously with the anime series, and was serialized in the manga magazine Ciao beginning in 1996. In 1999, Be-Papas produced the film Adolescence of Utena as a follow-up to the television anime series. The series has had several iterations of physical release, including a remaster overseen by Ikuhara in 2008. In North America, Utena was initially distributed by Central Park Media starting in 1998; the license for the series has been held by Crunchyroll since its 2023 acquisition of Right Stuf and its subsidiary Nozomi Entertainment, which acquired the license for Utena in 2010.'
    #     input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    #     input_ids = torch.cat([input_ids, torch.zeros((1, 512-input_ids.shape[1]), dtype=torch.long).to('cuda')], dim=1)
    # elif size == 1024:
    #     text = "As a child, Utena Tenjou was given a rose-engraved signet ring by a traveling prince, who promised her that they would one day meet again. Inspired by the encounter, Utena vowed to one day 'become a prince' herself. Years later, a teenaged Utena is a student at Ohtori Academy, an exclusive boarding school. She finds herself drawn into a sword dueling tournament with the school's Student Council, whose members wear signet rings identical to her own. The duelists compete to win the hand of Anthy Himemiya, a mysterious student known as the 'Rose Bride' who is said to possess the 'power to revolutionize the world'. Utena emerges victorious in her first duel; obliged to defend her position as the Rose Bride's fiancée, she decides to remain in the tournament to protect Anthy from those who seek the power of the Rose Bride for themselves. After dueling and achieving victory over the council, Utena is confronted by Souji Mikage, a student prodigy who uses his powers of persuasion and knowledge of psychology to manipulate others into becoming duelists. Mikage aims to kill Anthy to install Mamiya Chida, a terminally ill boy, as the Rose Bride. Utena defeats each of Mikage's duelists, and ultimately Mikage himself. Following his defeat, Mikage vanishes from Ohtori Academy, and the denizens of the school seemingly forget that he ever existed. It transpires that Akio Ohtori, the school's chairman and Anthy's brother, was using Mikage as part of a plot to obtain the 'power of eternity'. Mamiya was in truth a disguised Anthy, who assisted Akio in his manipulation of Mikage. Akio appears before each of the Student Council members, and takes them to a place he refers to as 'the end of the world'. Following their encounters with Akio, each of the Council members face Utena in rematches. Utena defeats the Council members once more, and is called to the dueling arena to meet the prince from her past. She discovers that the prince was Akio, and that he intends to use her and Anthy to gain the power of eternity for himself. Utena duels Akio to free Anthy from his influence; Anthy, complicit in her brother's scheme, intervenes and stabs Utena through the back. Akio attempts and fails to open the sealed gate that holds the power; a gravely injured Utena pries the gate open, where she discovers Anthy inside. Utena reaches out to her, and they briefly join hands as the dueling arena crumbles around them. Utena vanishes from Ohtori Academy, and all save for Akio and Anthy begin to forget her existence. Akio comments that Utena failed to bring about a revolution, and that he intends to begin a new attempt to attain the power of eternity; Anthy responds that Utena has merely left Ohtori Academy, and that she intends to do the same. Anthy solemnly vows to find Utena, and departs from Akio and the school."
    #     input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    #     input_ids = torch.cat([input_ids, torch.zeros((1, 1024-input_ids.shape[1]), dtype=torch.long).to('cuda')], dim=1)
    # elif size == 2048:
    #     text = "As a child, Utena Tenjou was given a rose-engraved signet ring by a traveling prince, who promised her that they would one day meet again. Inspired by the encounter, Utena vowed to one day 'become a prince' herself. Years later, a teenaged Utena is a student at Ohtori Academy, an exclusive boarding school. She finds herself drawn into a sword dueling tournament with the school's Student Council, whose members wear signet rings identical to her own. The duelists compete to win the hand of Anthy Himemiya, a mysterious student known as the 'Rose Bride' who is said to possess the 'power to revolutionize the world'. Utena emerges victorious in her first duel; obliged to defend her position as the Rose Bride's fiancée, she decides to remain in the tournament to protect Anthy from those who seek the power of the Rose Bride for themselves. After dueling and achieving victory over the council, Utena is confronted by Souji Mikage, a student prodigy who uses his powers of persuasion and knowledge of psychology to manipulate others into becoming duelists. Mikage aims to kill Anthy to install Mamiya Chida, a terminally ill boy, as the Rose Bride. Utena defeats each of Mikage's duelists, and ultimately Mikage himself. Following his defeat, Mikage vanishes from Ohtori Academy, and the denizens of the school seemingly forget that he ever existed. It transpires that Akio Ohtori, the school's chairman and Anthy's brother, was using Mikage as part of a plot to obtain the 'power of eternity'. Mamiya was in truth a disguised Anthy, who assisted Akio in his manipulation of Mikage. Akio appears before each of the Student Council members, and takes them to a place he refers to as 'the end of the world'. Following their encounters with Akio, each of the Council members face Utena in rematches. Utena defeats the Council members once more, and is called to the dueling arena to meet the prince from her past. She discovers that the prince was Akio, and that he intends to use her and Anthy to gain the power of eternity for himself. Utena duels Akio to free Anthy from his influence; Anthy, complicit in her brother's scheme, intervenes and stabs Utena through the back. Akio attempts and fails to open the sealed gate that holds the power; a gravely injured Utena pries the gate open, where she discovers Anthy inside. Utena reaches out to her, and they briefly join hands as the dueling arena crumbles around them. Utena vanishes from Ohtori Academy, and all save for Akio and Anthy begin to forget her existence. Akio comments that Utena failed to bring about a revolution, and that he intends to begin a new attempt to attain the power of eternity; Anthy responds that Utena has merely left Ohtori Academy, and that she intends to do the same. Anthy solemnly vows to find Utena, and departs from Akio and the school. As a child, Utena Tenjou was given a rose-engraved signet ring by a traveling prince, who promised her that they would one day meet again. Inspired by the encounter, Utena vowed to one day 'become a prince' herself. Years later, a teenaged Utena is a student at Ohtori Academy, an exclusive boarding school. She finds herself drawn into a sword dueling tournament with the school's Student Council, whose members wear signet rings identical to her own. The duelists compete to win the hand of Anthy Himemiya, a mysterious student known as the 'Rose Bride' who is said to possess the 'power to revolutionize the world'. Utena emerges victorious in her first duel; obliged to defend her position as the Rose Bride's fiancée, she decides to remain in the tournament to protect Anthy from those who seek the power of the Rose Bride for themselves. After dueling and achieving victory over the council, Utena is confronted by Souji Mikage, a student prodigy who uses his powers of persuasion and knowledge of psychology to manipulate others into becoming duelists. Mikage aims to kill Anthy to install Mamiya Chida, a terminally ill boy, as the Rose Bride. Utena defeats each of Mikage's duelists, and ultimately Mikage himself. Following his defeat, Mikage vanishes from Ohtori Academy, and the denizens of the school seemingly forget that he ever existed. It transpires that Akio Ohtori, the school's chairman and Anthy's brother, was using Mikage as part of a plot to obtain the 'power of eternity'. Mamiya was in truth a disguised Anthy, who assisted Akio in his manipulation of Mikage. Akio appears before each of the Student Council members, and takes them to a place he refers to as 'the end of the world'. Following their encounters with Akio, each of the Council members face Utena in rematches. Utena defeats the Council members once more, and is called to the dueling arena to meet the prince from her past. She discovers that the prince was Akio, and that he intends to use her and Anthy to gain the power of eternity for himself. Utena duels Akio to free Anthy from his influence; Anthy, complicit in her brother's scheme, intervenes and stabs Utena through the back. Akio attempts and fails to open the sealed gate that holds the power; a gravely injured Utena pries the gate open, where she discovers Anthy inside. Utena reaches out to her, and they briefly join hands as the dueling arena crumbles around them. Utena vanishes from Ohtori Academy, and all save for Akio and Anthy begin to forget her existence. Akio comments that Utena failed to bring about a revolution, and that he intends to begin a new attempt to attain the power of eternity; Anthy responds that Utena has merely left Ohtori Academy, and that she intends to do the same. Anthy solemnly vows to find Utena, and departs from Akio and the school."
    #     input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    #     input_ids = torch.cat([input_ids, torch.zeros((1, 2048-input_ids.shape[1]), dtype=torch.long).to('cuda')], dim=1)

    text = '''Earn monthly interest on our Citibank Time Deposits (also known as Fixed Deposits). What's more, you can get to enjoy the flexibility of making partial withdrawals before maturity date of your Time Deposit. Partial withdrawals in multiples of the'''
    input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    
    t1 = time.time()
    with torch.no_grad():
        # for i in range(100):
        #     model(input_ids, head_z=head_z, intermediate_z=intermediate_z, hidden_z=hidden_z)
        # generate text
        model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            max_length=size,
            head_z=head_z,
            intermediate_z=intermediate_z,
            hidden_z=hidden_z,
        )
    t2 = time.time()
    print(f'{(t2-t1)/100}')

def model_sparsity(hidden_z, head_z, intermediate_z):
    head_nums = np.outer(head_z.reshape(-1), hidden_z).sum().item() # 
    intermediate_nums = np.outer(intermediate_z.reshape(-1), hidden_z).sum().item()
    remaining_model_size = head_nums * (4096 // 32) * 4 + intermediate_nums * 3

    MODEL_SIZE = 6476005376
    # print('sparsity:', (MODEL_SIZE-remaining_model_size)/MODEL_SIZE)


if __name__ == '__main__':
    pefts = [
        'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_layerdistill_16bs/epoch4',
        'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_distill/epoch1',
        'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/20k_c4_2epoch_supervised/epoch1',
        'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_distill_full_hidden/epoch4',
        'llm_pruner'
    ]
    MODEL_NAME = 'baffo32/decapoda-research-llama-7B-hf'
    full_model, tokenizer = load_model(MODEL_NAME)

    for size in [256, 512, 1024, 2048]:
        print(f'length: {size}')
        model_latency(full_model, tokenizer, size)
        for peft in pefts:
            hidden_z, head_z, intermediate_z = load_zs(peft)
            model_sparsity(hidden_z > 0, head_z > 0, intermediate_z > 0)
            model_latency(full_model, tokenizer, size, (hidden_z, head_z, intermediate_z))


