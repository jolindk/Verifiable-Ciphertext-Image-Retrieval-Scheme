from homomorphic_encryption.paillier import Paillier


class EncryptBitmap:

    def __init__(self):

        self.paillier = Paillier()

        self.pub, self.priv = self.paillier.keygen()


    def encrypt_index(self, bitmap_index):

        encrypted_index = []

        for bitmap in bitmap_index:

            enc_bitmap = []

            for bit in bitmap:

                c = self.paillier.encrypt(self.pub, bit)

                enc_bitmap.append(c)

            encrypted_index.append(enc_bitmap)

        return encrypted_index


    def decrypt_bitmap(self, encrypted_bitmap):

        dec_bitmap = []

        for c in encrypted_bitmap:

            m = self.paillier.decrypt(self.pub, self.priv, c)

            dec_bitmap.append(m)

        return dec_bitmap
